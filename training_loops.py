from diffusers import StableDiffusionPipeline, SchedulerMixin
import torch
import wandb
import torch.nn.functional as F
from data_helpers import make_dataloader
from static_globals import *
import gc
from inference import call_vanilla_with_dict
from random import sample

def loop_vanilla(images: list,
               text_prompt_list:list,
               validation_prompt_list:list,
               pipeline:StableDiffusionPipeline,
               start_epoch:int,
               accelerator:object,
               epochs:int,
               seed:int,
                num_inference_steps:int,
                num_validation_images:int,
                noise_offset:float,
                batch_size:int,
                size:int,
                token_dict:dict={}
               )->StableDiffusionPipeline:
    '''
    anilla normal textual inversion training
    '''
    print(f"begin training method  vanilla on device {accelerator.device}")
    tracker=accelerator.get_tracker("wandb")
    for i in range(num_validation_images):
        wandb.define_metric(f"vanilla_img_{i}",step_metric="custom_step")
    tokenizer=pipeline.tokenizer
    vae=pipeline.vae
    text_encoder=pipeline.text_encoder
    unet=pipeline.unet
    scheduler=pipeline.scheduler
    dataloader=make_dataloader(images,text_prompt_list,size,batch_size,tokenizer)
    print("len dataloader",len(dataloader))
    print("len images ",len(images))
    print("len text prompt list",len(text_prompt_list))
    unet=pipeline.unet
    optimizer=torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    
    unet,text_encoder,vae,tokenizer, optimizer, dataloader, scheduler= accelerator.prepare(
        unet,text_encoder,vae,tokenizer, optimizer, dataloader, scheduler
    )
    added_cond_kwargs={}
    weight_dtype=pipeline.dtype
    global_step=0
    for e in range(start_epoch, epochs):
        train_loss = 0.0
        for step,batch in enumerate(dataloader):
            batch_size=batch[IMAGES].shape[0]
            print(f"batch size {batch_size}")
            device=accelerator.device
            with accelerator.accumulate(unet,text_encoder):
                latents = vae.encode(batch[IMAGES].to(dtype=weight_dtype)).latent_dist.sample().to(device=device)
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if noise_offset:
                    noise += noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=device
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
                timesteps = timesteps.long()
                if timesteps in token_dict:
                    print(timesteps.long().detach().tolist())
                    placeholder=token_dict[timesteps.long().detach().tolist()]
                else:
                    placeholder=PLACEHOLDER
                text=sample(text_prompt_list,bsz)
                text=[t.format(placeholder) for t in text]
                input_ids=tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(device=accelerator.device)
                encoder_hidden_states=text_encoder(input_ids)[0]
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                #encoder_hidden_states = text_encoder(batch[TEXT_INPUT_IDS])[0]
                #print('text_encoder(batch[TEXT_INPUT_IDS])',text_encoder(batch[TEXT_INPUT_IDS]))
                #print('encoder_hidden_states.size()',encoder_hidden_states.size())

                noise_pred = unet(noisy_latents, 
                                timesteps, 
                                encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs).sample
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item()
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({f"vanilla_train_loss": train_loss})
                train_loss = 0.0
        if accelerator.is_main_process:

            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(seed)
            path=f"vanilla_tmp.png"
            for i in range(num_validation_images):
                val_prompt=validation_prompt_list[i %len(validation_prompt_list)]
                print(f"validation vanilla_img_{i} {val_prompt} saved at {path}")
                added_cond_kwargs={}
                if len(token_dict)>0:
                    img=call_vanilla_with_dict(pipeline,prompt=val_prompt,
                                               num_inference_steps=num_inference_steps, generator=generator,safety_checker=None,token_dict=token_dict).images[0]
                else:
                    val_prompt=val_prompt.format(PLACEHOLDER)
                    img=pipeline(val_prompt, num_inference_steps=num_inference_steps, generator=generator,safety_checker=None).images[0]
                img.save(path)
                tracker.log({f"vanilla_img_{i}": wandb.Image(path)})
    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()
    return pipeline
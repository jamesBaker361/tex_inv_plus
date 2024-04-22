from diffusers import StableDiffusionPipeline, SchedulerMixin
import torch
import wandb
import torch.nn.functional as F
from data_helpers import make_dataloader
from static_globals import *
import gc
from inference import call_vanilla_with_dict
from random import sample
from custom_pipelines import T5UnetPipeline
import random
from gpu import print_details

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
                token_dict:dict={},
                prior:bool=False,
                prior_class:str="",
                lr:float=0.04
               )->StableDiffusionPipeline:
    '''
    anilla normal textual inversion training
    '''
    print(f"begin training method  vanilla on device {accelerator.device}")
    print(token_dict)
    tracker=accelerator.get_tracker("wandb")
    for i in range(num_validation_images):
        wandb.define_metric(f"vanilla_img_{i}",step_metric="custom_step")
    tokenizer=pipeline.tokenizer
    vae=pipeline.vae
    text_encoder=pipeline.text_encoder
    unet=pipeline.unet
    scheduler=pipeline.scheduler
    if prior:
        prior_images=[
            pipeline(prior_class,num_inference_steps=num_inference_steps, safety_checker=None).images[0] for _ in images
        ]
        prior_text_prompt_list=[
            prior_class for _ in images
        ]
        dataloader=make_dataloader(images,text_prompt_list,size,batch_size,tokenizer,prior_images,prior_text_prompt_list)
    else:
        dataloader=make_dataloader(images,text_prompt_list,size,batch_size,tokenizer)
    print("len dataloader",len(dataloader))
    print("len images ",len(images))
    print("len text prompt list",len(text_prompt_list))
    unet=pipeline.unet
    optimizer=torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=lr,
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
    device=accelerator.device
    for e in range(start_epoch, epochs):
        train_loss = 0.0
        print("training loops line 75")
        print_details()
        for step,batch in enumerate(dataloader):
            batch_size=batch[IMAGES].shape[0]
            with accelerator.accumulate(text_encoder):
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
                if len(token_dict)>0:
                    time_key_list=[k for k in token_dict.keys()]
                    timesteps_array=random.choices(time_key_list,k=bsz)
                    #timesteps = torch.randint(0, num_inference_steps, (latents.shape[0],), device=device)
                    timesteps=torch.tensor(timesteps_array,device=device)
                else: 
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                timesteps = timesteps.long()
                text=sample(text_prompt_list,bsz)
                if len(token_dict)>0:
                    for count,time_key in enumerate(timesteps.long().detach().tolist()):
                        text[count]=text[count].format(token_dict[time_key])
                else:
                    placeholder=PLACEHOLDER
                    text=[t.format(placeholder) for t in text]
                #print('loop vanilal text',text)
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


                noise_pred = unet(noisy_latents, 
                                timesteps, 
                                encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs).sample
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                if prior:
                    prior_latents=vae.encode(batch[PRIOR_IMAGES].to(dtype=weight_dtype)).latent_dist.sample().to(device=device)* vae.config.scaling_factor
                    prior_noise = torch.randn_like(latents)
                    if noise_offset:
                        prior_noise += noise_offset * torch.randn(
                            (prior_latents.shape[0], prior_latents.shape[1], 1, 1), device=device
                        )
                    prior_noisy_latents=scheduler.add_noise(prior_latents, prior_noise, timesteps)
                    prior_encoder_hidden_states=text_encoder(batch[PRIOR_TEXT_INPUT_IDS])[0].to(device).to(weight_dtype)
                    prior_noise_pred=unet(prior_noisy_latents,
                                          timesteps,
                                          prior_encoder_hidden_states,
                                          added_cond_kwargs=added_cond_kwargs).sample
                    
                    prior_loss = F.mse_loss(prior_noise_pred.float(), prior_noise.float(), reduction="mean")
                    loss=loss+prior_loss

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item()
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                params_to_clip =[p for p in text_encoder.parameters()]
                accelerator.clip_grad_norm_(params_to_clip, 1.0)
                global_step += 1
                accelerator.log({f"train_loss": train_loss})
                train_loss = 0.0
        '''if accelerator.is_main_process:

            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(seed)
            path=f"vanilla_tmp.png"
            for i in range(num_validation_images):
                val_prompt=validation_prompt_list[i %len(validation_prompt_list)]
                print(f"validation vanilla_img_{i} {val_prompt} saved at {path}")
                added_cond_kwargs={}
                if len(token_dict)>0:
                    img=call_vanilla_with_dict(pipeline,prompt=val_prompt,
                                               num_inference_steps=num_inference_steps, 
                                               generator=generator,safety_checker=None,token_dict=token_dict,
                                               #timesteps=[k for k in token_dict.keys()]
                                               ).images[0]
                else:
                    val_prompt=val_prompt.format(PLACEHOLDER)
                    img=pipeline(val_prompt, num_inference_steps=num_inference_steps, generator=generator,safety_checker=None).images[0]
                img.save(path)
                tracker.log({f"vanilla_img_{i}": wandb.Image(path)})'''
        accelerator.free_memory()
        torch.cuda.empty_cache()
        gc.collect()
    return pipeline

def loop_general(images: list,
               text_prompt_list:list,
               validation_prompt_list:list,
               pipeline:T5UnetPipeline,
               start_epoch:int,
               accelerator:object,
               epochs:int,
               seed:int,
                num_inference_steps:int,
                num_validation_images:int,
                noise_offset:float,
                batch_size:int,
                size:int,
                training_method:str,
                token_dict:dict={},
                train_adapter:bool=False,
                lr:float=0.04):
    print(f"begin training method  {training_method} on device {accelerator.device}")
    #third_image=pipeline("thing")[0]
    #third_image.save(f"{training_method}_third.png")
    print(token_dict)
    tracker=accelerator.get_tracker("wandb")
    for i in range(num_validation_images):
        wandb.define_metric(f"{training_method}_img_{i}",step_metric="custom_step")
    if training_method==T5_UNET or training_method==LLAMA_UNET:
        max_length=77
    elif training_method==T5_TRANSFORMER:
        max_length=120
    pipeline.prepare(accelerator)
    #fourth_image=pipeline("thing")[0]
    #fourth_image.save(f"{training_method}_fourth.png")
    text_encoder=pipeline.text_encoder
    tokenizer=pipeline.tokenizer
    vae=pipeline.vae
    scheduler=pipeline.scheduler
    vis=pipeline.vis
    adapter=pipeline.adapter
    adapter.requires_grad_(train_adapter)
    trainable_params=[p for p in text_encoder.get_input_embeddings().parameters()]+[p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    dataloader=make_dataloader(images,text_prompt_list,size,batch_size,tokenizer)
    optimizer,dataloader=accelerator.prepare(
        optimizer,dataloader
    )

    added_cond_kwargs={}
    weight_dtype=vae.dtype
    global_step=0
    device=accelerator.device
    generator = torch.Generator()
    generator.manual_seed(seed)
    path=f"{training_method}_tmp.png"
    for i in range(num_validation_images):
        val_prompt=validation_prompt_list[i %len(validation_prompt_list)]
        print(f"validation {training_method}_img_{i} {val_prompt} saved at {path}")
        added_cond_kwargs={}
        if len(token_dict)>0:
            img=pipeline(val_prompt,
                                    num_inference_steps=num_inference_steps, 
                                    #generator=generator,
                                    token_dict=token_dict)[0]
        else:
            val_prompt=val_prompt.format(PLACEHOLDER)
            print(val_prompt)
            img=pipeline(val_prompt, num_inference_steps=num_inference_steps, 
                         #generator=generator
                         )[0]
        img.save(path)
        tracker.log({f"{training_method}_{i}": wandb.Image(path)})
    trainable_models=[text_encoder]
    if train_adapter:
        trainable_models.append(adapter)
    for e in range(start_epoch, epochs):
        train_loss = 0.0
        for step,batch in enumerate(dataloader):
            with accelerator.accumulate(*trainable_models):
                # Latent preparation
                latents = vae.encode(batch[IMAGES].to(dtype=weight_dtype)).latent_dist.sample().to(device=device)
                latents = latents * 0.18215
                noise = torch.randn_like(latents)
                if noise_offset:
                        noise += noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=device
                        )
                bsz = latents.shape[0]
                if len(token_dict)>0:
                    time_key_list=[k for k in token_dict.keys()]
                    timesteps_array=random.choices(time_key_list,k=bsz)
                    #timesteps = torch.randint(0, num_inference_steps, (latents.shape[0],), device=device)
                    timesteps=torch.tensor(timesteps_array,device=device)
                else: 
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                timesteps = timesteps.long()

                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                text=sample(text_prompt_list,bsz)
                if len(token_dict)>0:
                    for count,time_key in enumerate(timesteps.long().detach().tolist()):
                        text[count]=text[count].format(token_dict[time_key])
                else:
                    placeholder=PLACEHOLDER
                    text=[t.format(placeholder) for t in text]
                #print('loop general text',text)
                if training_method==T5_UNET or training_method==LLAMA_UNET:
                    text_input = tokenizer(
                        text, 
                        padding="max_length", 
                        max_length=max_length, 
                        return_tensors="pt", 
                        truncation=True,
                    ).input_ids.to(device)
                elif training_method==T5_TRANSFORMER:
                    text_input = tokenizer(
                        text, 
                        padding="max_length", 
                        max_length=max_length, 
                        add_special_tokens=True,
                        return_tensors="pt", 
                        truncation=True,
                    ).to(device)
                if training_method==T5_UNET:
                    encoder_hidden_states_pre = text_encoder(text_input)[0]
                    encoder_hidden_states = adapter(encoder_hidden_states_pre).sample
                elif training_method==T5_TRANSFORMER:
                    prompt_attention_mask = text_input.attention_mask
                    encoder_hidden_states_pre = text_encoder(text_input.input_ids, attention_mask=prompt_attention_mask)[0]
                    encoder_hidden_states = adapter(encoder_hidden_states_pre).sample
                elif training_method==LLAMA_UNET:
                    encoder_hidden_states_pre = text_encoder(text_input, output_hidden_states=True).hidden_states[-1]
                    encoder_hidden_states = adapter(encoder_hidden_states_pre).sample

                if training_method==T5_TRANSFORMER:
                    model_pred=vis(
                    noisy_latents, 
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timesteps, 
                    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                    ).sample
                    model_pred = model_pred.chunk(2, dim=1)[0]
                else:
                    model_pred = vis(noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")            
                accelerator.backward(loss)

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item()

                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    params_to_clip =[p for p in text_encoder.parameters()]+[p for p in adapter.parameters()]
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    global_step += 1
                    accelerator.log({f"train_loss": train_loss})
                    train_loss = 0.0
        '''if accelerator.is_main_process:
            generator = torch.Generator()
            generator.manual_seed(seed)
            path=f"{training_method}_tmp.png"
            for i in range(num_validation_images):
                val_prompt=validation_prompt_list[i %len(validation_prompt_list)]
                print(f"validation {training_method}_img_{i} {val_prompt} saved at {path}")
                added_cond_kwargs={}
                if len(token_dict)>0:
                    img=pipeline(val_prompt,
                                            num_inference_steps=num_inference_steps, 
                                            #generator=generator,
                                            token_dict=token_dict)[0]
                else:
                    val_prompt=val_prompt.format(PLACEHOLDER)
                    img=pipeline(val_prompt, num_inference_steps=num_inference_steps, 
                                 #generator=generator
                                 )[0]
                img.save(path)
                tracker.log({f"{training_method}_{i}": wandb.Image(path)})'''
        accelerator.free_memory()
        torch.cuda.empty_cache()
        gc.collect()
    return pipeline



'''def loop_t5_transformer(images: list,
               text_prompt_list:list,
               validation_prompt_list:list,
               pipeline:T5UnetPipeline,
               start_epoch:int,
               accelerator:object,
               epochs:int,
               seed:int,
                num_inference_steps:int,
                num_validation_images:int,
                noise_offset:float,
                batch_size:int,
                size:int,
                token_dict:dict={}):
    print(f"begin training method  t5 transformer on device {accelerator.device}")
    print(token_dict)
    tracker=accelerator.get_tracker("wandb")
    for i in range(num_validation_images):
        wandb.define_metric(f"t5_transformer_img_{i}",step_metric="custom_step")
    pipeline.prepare(accelerator)

    text_encoder=pipeline.text_encoder
    tokenizer=pipeline.tokenizer
    vae=pipeline.vae
    scheduler=pipeline.scheduler
    vis=pipeline.vis
    adapter=pipeline.adapter
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    dataloader=make_dataloader(images,text_prompt_list,size,batch_size,tokenizer)
    optimizer,dataloader=accelerator.prepare(optimizer,dataloader)
    weight_dtype=vae.dtype
    global_step=0
    device=accelerator.device
    for e in range(start_epoch, epochs):
        train_loss = 0.0
        for step,batch in enumerate(dataloader):
            with accelerator.accumulate(vis,text_encoder):
                latents = vae.encode(batch[IMAGES].to(dtype=weight_dtype)).latent_dist.sample().to(device=device)
                latents = latents * 0.18215
                noise = torch.randn_like(latents)
                if noise_offset:
                        noise += noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=device
                        )
                bsz = latents.shape[0]
                if len(token_dict)>0:
                    time_key_list=[k for k in token_dict.keys()]
                    timesteps_array=random.choices(time_key_list,k=bsz)
                    #timesteps = torch.randint(0, num_inference_steps, (latents.shape[0],), device=device)
                    timesteps=torch.tensor(timesteps_array,device=device)
                else: 
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                timesteps = timesteps.long()

                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                text=sample(text_prompt_list,bsz)
                if len(token_dict)>0:
                    for count,time_key in enumerate(timesteps.long().detach().tolist()):
                        text[count]=text[count].format(token_dict[time_key])
                else:
                    placeholder=PLACEHOLDER
                    text=[t.format(placeholder) for t in text]
                text_inputs = tokenizer(
                batch[1], 
                    padding="max_length", 
                    max_length=120, 
                    add_special_tokens=True, 
                    return_tensors="pt",
                    truncation=True, 
                ).to(accelerator.device)
                prompt_attention_mask = text_inputs.attention_mask
                encoder_hidden_states_pre = text_encoder(text_inputs.input_ids, attention_mask=prompt_attention_mask)[0]
                encoder_hidden_states = adapter(encoder_hidden_states_pre).sample
                '''
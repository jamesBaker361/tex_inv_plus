from accelerate import Accelerator
import os
import sys
sys.path.append("../")
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
sys.path.append("../")
sys.path.append("/home/jlb638/Desktop/prompt")
sys.path.append("/home/jlb638/Desktop/prompt/lavi")

import argparse
from tqdm.auto import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download,snapshot_download

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler,Transformer2DModel
from transformers import AutoTokenizer, T5EncoderModel,LlamaForCausalLM, LlamaTokenizer

from modules.lora import monkeypatch_or_replace_lora_extended
from modules.adapters import TextAdapter
from gpu import print_details

class PreparePipeline:
    def prepare(self,accelerator:Accelerator):
        self.vae, self.vis,self.text_encoder,self.adapter,self.noise_scheduler=accelerator.prepare(
            self.vae, self.vis,self.text_encoder,self.adapter,self.noise_scheduler
        )


class T5UnetPipeline(PreparePipeline):
    def __init__(self):
        VIS_REPLACE_MODULES = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}
        TEXT_ENCODER_REPLACE_MODULES = {"T5Attention"}
        # Modules of T2I diffusion models
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        self.vis = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.noise_scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-large")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-large")
        checkpoint_dir=snapshot_download("shihaozhao/LaVi-Bridge")
        self.adapter = TextAdapter.from_pretrained(os.path.join(checkpoint_dir, f"t5_unet/adapter"))

        # LoRA
        monkeypatch_or_replace_lora_extended(
            self.vis, 
            torch.load(os.path.join(checkpoint_dir, f"t5_unet/lora_vis.pt")), 
            r=32, 
            target_replace_module=VIS_REPLACE_MODULES,
        )
        monkeypatch_or_replace_lora_extended(
            self.text_encoder, 
            torch.load(os.path.join(checkpoint_dir, f"t5_unet/lora_text.pt")), 
            r=32, 
            target_replace_module=TEXT_ENCODER_REPLACE_MODULES,
        )

        for model in [self.vae, self.vis,self.text_encoder,self.adapter]:
            model.requires_grad_(False)
    
    def __call__(self,prompts,
                 negative_prompts=None,
                 num_inference_steps:int=30,
                 guidance_scale:float=7.5,
                 size:int=512,
                 token_dict:dict={}
                 )->list:
        if type(prompts)==type("string"):
            prompts==[prompts]
        if negative_prompts!=None and type(negative_prompts)==type("string"):
            negative_prompts=[negative_prompts]
        if negative_prompts!=None and len(negative_prompts)==1:
            negative_prompts=negative_prompts*len(prompts)
        if negative_prompts!=None and len(negative_prompts)!=len(prompts):
            raise Exception(f"mismatch between negative prompts len {len(negative_prompts)} and prompts len {len(prompts)}")
        torch_device=self.vis.device
        images=[]
        with torch.no_grad():
            for k,prompt in enumerate(prompts):
                text_ids = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt", truncation=True).input_ids.to(torch_device)
                text_embeddings = self.text_encoder(input_ids=text_ids)[0]
                text_embeddings = self.adapter(text_embeddings).sample
                if negative_prompts==None:
                    uncond_input = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
                else:
                    uncond_input = self.tokenizer(negative_prompts[k], padding="max_length", max_length=77, return_tensors="pt")
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(torch_device))[0]
                uncond_embeddings =  self.adapter(uncond_embeddings).sample
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                # Latent preparation
                latents = torch.randn((1, self.vis.in_channels, size // 8, size // 8)).to(torch_device)
                latents = latents * self.noise_scheduler.init_noise_sigma

                # Model prediction
                self.noise_scheduler.set_timesteps(num_inference_steps)
                for t in tqdm(self.noise_scheduler.timesteps):
                    timestep_key=t.long().detach().tolist()
                    if timestep_key in token_dict:
                        placeholder=token_dict[timestep_key]
                        text_ids = self.tokenizer(prompt.format(placeholder), padding="max_length", max_length=77, return_tensors="pt", truncation=True).input_ids.to(torch_device)
                        text_embeddings = self.text_encoder(input_ids=text_ids)[0]
                        text_embeddings = self.adapter(text_embeddings).sample
                        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                    noise_pred = self.vis(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

                # Decoding
                latents = 1 / 0.18215 * latents
                image = self.vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1).squeeze()
                image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                image = Image.fromarray(image)
                images.append(image)
        return images
    
class T5TransformerPipeline(PreparePipeline):
    def __init__(self):
        VIS_REPLACE_MODULES = {"Attention", "GEGLU"}
        TEXT_ENCODER_REPLACE_MODULES = {"T5Attention"}
        # Modules of T2I diffusion models
        self.vae = AutoencoderKL.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", subfolder="vae")
        self.vis = Transformer2DModel.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", subfolder="transformer")
        self.noise_scheduler = UniPCMultistepScheduler.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", subfolder="scheduler")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-large")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-large")
        checkpoint_dir=snapshot_download("shihaozhao/LaVi-Bridge")
        self.adapter = TextAdapter.from_pretrained(os.path.join(checkpoint_dir, f"t5_transformer/adapter"), use_safetensors=True)

        # LoRA
        monkeypatch_or_replace_lora_extended(
            self.vis, 
            torch.load(os.path.join(checkpoint_dir, f"t5_transformer/lora_vis.pt")), 
            r=32, 
            target_replace_module=VIS_REPLACE_MODULES,
        )
        monkeypatch_or_replace_lora_extended(
            self.text_encoder, 
            torch.load(os.path.join(checkpoint_dir, f"t5_transformer/lora_text.pt")), 
            r=32, 
            target_replace_module=TEXT_ENCODER_REPLACE_MODULES,
        )

        for model in [self.vae, self.vis,self.text_encoder,self.adapter]:
            model.requires_grad_(False)

    def __call__(self,prompts,
                 negative_prompts=None,
                 num_inference_steps:int=30,
                 guidance_scale:float=7.5,
                 size:int=512,
                 token_dict:dict={}
                 )->list:
        images=[]
        if type(prompts)==type("string"):
            prompts==[prompts]
        if negative_prompts!=None and type(negative_prompts)==type("string"):
            negative_prompts=[negative_prompts]
        if negative_prompts!=None and len(negative_prompts)==1:
            negative_prompts=negative_prompts*len(prompts)
        if negative_prompts!=None and len(negative_prompts)!=len(prompts):
            raise Exception(f"mismatch between negative prompts len {len(negative_prompts)} and prompts len {len(prompts)}")
        torch_device=self.vis.device
        with torch.no_grad():
            for k,prompt in enumerate(prompts):
                # Text embeddings

                text_inputs = self.tokenizer(prompt, padding="max_length", max_length=120, add_special_tokens=True, return_tensors="pt", truncation=True).to(torch_device)
                text_input_ids = text_inputs.input_ids
                prompt_attention_mask = text_inputs.attention_mask
                encoder_hidden_states = self.text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
                encoder_hidden_states = self.adapter(encoder_hidden_states).sample
                if negative_prompts==None:
                    neg_text_inputs = self.tokenizer([""], padding="max_length", max_length=120, add_special_tokens=True, return_tensors="pt", truncation=True).to(torch_device)
                else:
                    neg_text_inputs = self.tokenizer(negative_prompts[k], padding="max_length", max_length=120, add_special_tokens=True, return_tensors="pt", truncation=True).to(torch_device)
                neg_text_input_ids = neg_text_inputs.input_ids
                neg_prompt_attention_mask = neg_text_inputs.attention_mask
                neg_encoder_hidden_states = self.text_encoder(neg_text_input_ids, attention_mask=neg_prompt_attention_mask)[0]
                neg_encoder_hidden_states = self.adapter(neg_encoder_hidden_states).sample
                text_embeddings = torch.cat([neg_encoder_hidden_states, encoder_hidden_states])
                attention_mask = torch.cat([neg_prompt_attention_mask, prompt_attention_mask])

                # Latent preparation
                latents = torch.randn((1, self.vis.in_channels, size // 8, size // 8)).to(torch_device)
                latents = latents * self.noise_scheduler.init_noise_sigma

                # Model prediction
                self.noise_scheduler.set_timesteps(num_inference_steps)
                for t in tqdm(self.noise_scheduler.timesteps):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                    noise_pred = self.vis(
                        latent_model_input, 
                        encoder_hidden_states=text_embeddings,
                        encoder_attention_mask=attention_mask,
                        timestep=torch.Tensor([t.item()]).expand(latent_model_input.shape[0]).to(torch_device),
                        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                    ).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                    latents = self.noise_scheduler.step(noise_pred, t, latents)[0]

                # Decoding
                latents = 1 / 0.18215 * latents
                image = self.vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1).squeeze()
                image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                image = Image.fromarray(image)
                images.append(image)
        return images
    

class LlamaUnetPipeline(PreparePipeline):
    def __init__(self,llama_dir="meta-llama/Llama-2-7b-hf"):
        VIS_REPLACE_MODULES = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}
        TEXT_ENCODER_REPLACE_MODULES = {"LlamaAttention"}
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",torch_dtype=torch.float16)
        self.vis = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",torch_dtype=torch.float16)
        self.noise_scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler",torch_dtype=torch.float16)
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_dir,torch_dtype=torch.float16)
        # To perform inference on a 24GB GPU memory, llama2 was converted to half precision
        self.text_encoder = LlamaForCausalLM.from_pretrained(llama_dir,torch_dtype=torch.float16)
        checkpoint_dir=snapshot_download("shihaozhao/LaVi-Bridge")
        #adapter_path=hf_hub_download("shihaozhao/LaVi-Bridge",subfolder=args.subfolder,filename="adapter")
        self.adapter = TextAdapter.from_pretrained(os.path.join(checkpoint_dir, f"llama2_unet/adapter"),torch_dtype=torch.float16)
        self.tokenizer.pad_token = '[PAD]'

        monkeypatch_or_replace_lora_extended(
            self.vis, 
            torch.load(os.path.join(checkpoint_dir, f"llama2_unet/lora_vis.pt")), 
            r=32, 
            target_replace_module=VIS_REPLACE_MODULES,
        )
        monkeypatch_or_replace_lora_extended(
            self.text_encoder, 
            torch.load(os.path.join(checkpoint_dir, f"llama2_unet/lora_text.pt")), 
            r=32, 
            target_replace_module=TEXT_ENCODER_REPLACE_MODULES,
        )

        for model in [self.vae, self.vis,self.text_encoder,self.adapter]:
            model.requires_grad_(False)

    def __call__(self, prompts,
                 negative_prompts=None,
                 num_inference_steps:int=30,
                 guidance_scale:float=7.5,
                 size:int=512,
                 token_dict:dict={}
                 )->list:
        images=[]
        if type(prompts)==type("string"):
            prompts==[prompts]
        if negative_prompts!=None and type(negative_prompts)==type("string"):
            negative_prompts=[negative_prompts]
        if negative_prompts!=None and len(negative_prompts)==1:
            negative_prompts=negative_prompts*len(prompts)
        if negative_prompts!=None and len(negative_prompts)!=len(prompts):
            raise Exception(f"mismatch between negative prompts len {len(negative_prompts)} and prompts len {len(prompts)}")
        torch_device=self.vis.device
        with torch.no_grad():
            for k,prompt in enumerate(prompts):
                # Text embeddings
                text_ids = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt", truncation=True).input_ids.to(torch_device)
                text_embeddings = self.text_encoder(input_ids=text_ids, output_hidden_states=True).hidden_states[-1].to(torch.float16)
                text_embeddings = self.adapter(text_embeddings).sample
                if negative_prompts==None:
                    uncond_input = self.tokenizer(negative_prompts[k], padding="max_length", max_length=77, return_tensors="pt")
                else:
                    uncond_input = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
                # Convert the text embedding back to full precision
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(torch_device), output_hidden_states=True).hidden_states[-1]
                uncond_embeddings =  self.adapter(uncond_embeddings).sample
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                # Latent preparation
                latents = torch.randn((1, self.vis.in_channels, size // 8,size // 8)).to(torch_device).to(torch.float16)
                latents = latents * self.noise_scheduler.init_noise_sigma

                # Model prediction
                self.noise_scheduler.set_timesteps(num_inference_steps)
                for t in tqdm(self.noise_scheduler.timesteps):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                    noise_pred = self.vis(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

                # Decoding
                latents = 1 / 0.18215 * latents
                image = self.vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1).squeeze()
                image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                image = Image.fromarray(image)
                images.append(image)
        return images
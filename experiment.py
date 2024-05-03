from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler,DPMSolverMultistepScheduler,DDPMScheduler,DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from accelerate import Accelerator
from static_globals import *
import torch
import random
import string
import nltk
import os
nltk.download('words')
from nltk.corpus import words
from training_loops import loop_vanilla,loop_general
from inference import call_vanilla_with_dict
import random
import ImageReward as image_reward
import string
def generate_random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))
reward_cache="/scratch/jlb638/reward_symbolic/"+generate_random_string(10)
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel
import numpy as np
from numpy.linalg import norm
import gc
from aesthetic_reward import get_aesthetic_scorer
from custom_pipelines import T5UnetPipeline,T5TransformerPipeline,LlamaUnetPipeline


def cos_sim(vector_i,vector_j)->float:
    return np.dot(vector_i,vector_j)/(norm(vector_i)*norm(vector_j))

def is_real_word(word):
    return word.lower() in words.words()

def generate_random_string(length=3):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def prepare_textual_inversion(placeholder:str, tokenizer:object,text_encoder:object,initializer_token:str="thing"):
    placeholder_tokens=[placeholder]
    tokenizer.add_tokens(placeholder_tokens)
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    text_encoder.get_input_embeddings().requires_grad_(True)
    return tokenizer,text_encoder

def prepare_from_token_strategy(timesteps: torch.Tensor,token_strategy:str,tokenizer,text_encoder):
    token_set=set()
    token_dict={}
    #print("prepare_from_token_strategy",timesteps)
    #print(timesteps.detach())
    #print(timesteps.detach().tolist())
    print(f"tokenizer len {len(tokenizer)}")
    if token_strategy==MULTI:
        for t in timesteps.detach().tolist():
            new_token=generate_random_string()
            while is_real_word(new_token) or new_token in token_set:
                new_token=generate_random_string()
            token_set.add(new_token)
            #print(t)
            token_dict[t]="<{}>".format(new_token)
            tokenizer,text_encoder=prepare_textual_inversion("<{}>".format(new_token), tokenizer, text_encoder)
    elif token_strategy==HALF or token_strategy==THIRD:
        n_tokens={
            HALF:2,
            THIRD:3
        }[token_strategy]
        n_steps=len([t for t in timesteps.detach().tolist()])
        steps_per_token=n_steps//n_tokens
        current_count=0
        current_token=generate_random_string()
        new_token=generate_random_string()
        while is_real_word(new_token) or new_token in token_set:
            current_token=generate_random_string()
        token_set.add(current_token)
        for count,t in enumerate(timesteps.detach().tolist()):
            token_dict[t]="<{}>".format(current_token)
            tokenizer,text_encoder=prepare_textual_inversion("<{}>".format(current_token), tokenizer, text_encoder)
            current_count+=1
            if current_count>steps_per_token:
                current_token=generate_random_string()
                while is_real_word(new_token) or new_token in token_set:
                    current_token=generate_random_string()
                token_set.add(current_token)
                current_count=0
    elif token_strategy==DEFAULT:
        tokenizer,text_encoder=prepare_textual_inversion(PLACEHOLDER, tokenizer, text_encoder)
    print(f"prepare_from_token_strategy tokenizer len {len(tokenizer)}")
    return tokenizer,text_encoder,token_dict

def get_metric_dict(evaluation_prompt_list:list, evaluation_image_list:list,image_list:list):
    metric_dict={}
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_inputs=clip_processor(text=evaluation_prompt_list, images=evaluation_image_list+image_list, return_tensors="pt", padding=True)

    outputs = clip_model(**clip_inputs)
    src_image_n=len(image_list)
    text_embed_list=outputs.text_embeds.detach().numpy()
    image_embed_list=outputs.image_embeds.detach().numpy()[:-src_image_n]
    src_image_embed_list=outputs.image_embeds.detach().numpy()[-src_image_n:]
    ir_model=image_reward.load("/scratch/jlb638/reward-blob",med_config="/scratch/jlb638/ImageReward/med_config.json")

    identity_consistency_list=[]
    target_similarity_list=[]
    prompt_similarity_list=[]
    for i in range(len(image_embed_list)):
        image_embed=image_embed_list[i]
        text_embed=text_embed_list[i]
        for src_image_embed in src_image_embed_list:
            target_similarity_list.append(cos_sim(image_embed,src_image_embed))
        prompt_similarity_list.append(cos_sim(image_embed, text_embed))
        for j in range(i+1, len(image_embed_list)):
            #print(i,j)
            vector_j=image_embed_list[j]
            sim=cos_sim(image_embed,vector_j)
            identity_consistency_list.append(sim)

    metric_dict[IDENTITY_CONSISTENCY]=np.mean(identity_consistency_list)
    metric_dict[TARGET_SIMILARITY]=np.mean(target_similarity_list)
    metric_dict[PROMPT_SIMILARITY]=np.mean(prompt_similarity_list)
    #for evaluation_image,evaluation_prompt in zip(evaluation_image_list, evaluation_prompt_list):
    metric_dict[IMAGE_REWARD]=np.mean(
        [ir_model.score(evaluation_prompt,evaluation_image) for evaluation_prompt,evaluation_image in zip(evaluation_prompt_list, evaluation_image_list) ]
    )
    aesthetic_scorer=get_aesthetic_scorer()
    metric_dict[AESTHETIC_SCORE]=np.mean(
        [aesthetic_scorer(evaluation_image).cpu().numpy()[0] for evaluation_image in evaluation_image_list]
    )
    for metric in METRIC_LIST:
        if metric not in metric_dict:
            metric_dict[metric]=0.0
    return metric_dict


def train_and_evaluate_one_sample_vanilla(
        image_list:list,
        prompt_list:list,
        epochs:int,
        pretrained_vanilla:str,
        accelerator:Accelerator,
        num_inference_steps:int,
        token_strategy:str,
        validation_prompt_list:list,
        seed:int,
        num_validation_images:int,
        noise_offset:float,
        batch_size:int,
        size:int,
        evaluation_prompt_list:list,
        prior:bool,
        prior_class:str,
        lr:float,
        lr_scheduler_type:str,
                lr_warmup_steps:int,
                lr_num_cycles:int,
                max_grad_norm:float,
                scheduler_type:str,
                long_evaluation_prompt_list:list,
                negative_token:bool,
                spare_token:bool,
                spare_lambda:float
):
    pipeline=StableDiffusionPipeline.from_pretrained(pretrained_vanilla)
    text_encoder=pipeline.text_encoder
    tokenizer=pipeline.tokenizer
    unet=pipeline.unet
    vae=pipeline.vae
    scheduler={
            "UniPCMultistepScheduler":UniPCMultistepScheduler,
            "DPMSolverMultistepScheduler":DPMSolverMultistepScheduler,
            "DDPMScheduler":DDPMScheduler,
            "DDIMScheduler":DDIMScheduler
        }[scheduler_type]
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
    scheduler=pipeline.scheduler
    for model in [vae,unet,text_encoder]:
        model.requires_grad_(False)
    
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, accelerator.device, None)
    print("len tokenizer" ,len(tokenizer))
    tokenizer,text_encoder,token_dict=prepare_from_token_strategy(timesteps,token_strategy,tokenizer,text_encoder)
    if negative_token:
        tokenizer,text_encoder=prepare_textual_inversion(NEGATIVE_PLACEHOLDER,tokenizer,text_encoder)
    if spare_token:
        tokenizer,text_encoder=prepare_textual_inversion(SPARE_PLACEHOLDER,tokenizer,text_encoder)
    print("len tokenizer" ,len(tokenizer))
    text_encoder.gradient_checkpointing_enable()
    pipeline=loop_vanilla(
        image_list,
        prompt_list,
        validation_prompt_list,
        pipeline,0,
        accelerator, epochs,seed,
        num_inference_steps,
        num_validation_images,
        noise_offset,
        batch_size,
        size,
        token_dict,
        prior,
        prior_class,
        lr,
        lr_scheduler_type,
                lr_warmup_steps,
                lr_num_cycles,
                max_grad_norm,
                negative_token,
                spare_token,
                spare_lambda
    )
    split_evaluation_image_list=[]
    split_metric_dict={}
    long_metric_dict={}
    long_evaluation_image_list=[]
    split_long_metric_dict={}
    long_evaluation_image_list=[]
    split_long_evaluation_image_list=[]
    sample_token=PLACEHOLDER
    negative_prompt=NEGATIVE
    if negative_token:
        negative_prompt+=","+NEGATIVE_PLACEHOLDER
    if spare_token:
        sample_token=PLACEHOLDER+","+SPARE_PLACEHOLDER
    if token_strategy==DEFAULT:
        evaluation_image_list=[
            pipeline(evaluation_prompt.format(sample_token),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=negative_prompt,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
        if spare_token:
            split_evaluation_image_list+=[pipeline(evaluation_prompt.format(PLACEHOLDER),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=negative_prompt,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
            ]
            split_evaluation_image_list+=[pipeline(evaluation_prompt.format(SPARE_PLACEHOLDER),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=negative_prompt,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
            ]
            split_metric_dict=get_metric_dict(evaluation_prompt_list+evaluation_prompt_list
                                              , split_evaluation_image_list, image_list)
    else:
        evaluation_image_list=[
            call_vanilla_with_dict(pipeline,evaluation_prompt,num_inference_steps=num_inference_steps,
                    negative_prompt=negative_prompt,
                    safety_checker=None,token_dict=token_dict).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
    metric_dict=get_metric_dict(evaluation_prompt_list, evaluation_image_list, image_list)
    if len(long_evaluation_prompt_list)>0:
        if token_strategy==DEFAULT:
            long_evaluation_image_list=[
                pipeline(evaluation_prompt.format(sample_token),
                        num_inference_steps=num_inference_steps,
                        negative_prompt=negative_prompt,
                        safety_checker=None).images[0] for evaluation_prompt in long_evaluation_prompt_list
            ]
            if spare_token:
                split_long_evaluation_image_list+=[
                pipeline(evaluation_prompt.format(PLACEHOLDER),
                        num_inference_steps=num_inference_steps,
                        negative_prompt=negative_prompt,
                        safety_checker=None).images[0] for evaluation_prompt in long_evaluation_prompt_list]
                split_long_evaluation_image_list+=[
                pipeline(evaluation_prompt.format(SPARE_PLACEHOLDER),
                        num_inference_steps=num_inference_steps,
                        negative_prompt=negative_prompt,
                        safety_checker=None).images[0] for evaluation_prompt in long_evaluation_prompt_list]
                split_long_metric_dict=get_metric_dict(long_evaluation_prompt_list+long_evaluation_prompt_list
                                                       ,split_long_evaluation_image_list,image_list)
        else:
            long_evaluation_image_list=[
                call_vanilla_with_dict(pipeline,evaluation_prompt,num_inference_steps=num_inference_steps,
                        negative_prompt=negative_prompt,
                        safety_checker=None,token_dict=token_dict).images[0] for evaluation_prompt in long_evaluation_prompt_list
            ]
        long_metric_dict=get_metric_dict(long_evaluation_prompt_list, long_evaluation_image_list,image_list)
    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()
    del tokenizer,text_encoder,token_dict,timesteps, num_inference_steps,pipeline
    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()
    return None,metric_dict,long_metric_dict,split_metric_dict,split_long_metric_dict,evaluation_image_list,long_evaluation_image_list,split_evaluation_image_list,split_long_evaluation_image_list

def train_and_evaluate_one_sample(
        image_list:list,
        prompt_list:list,
        epochs:int,
        training_method:str,
        accelerator:Accelerator,
        num_inference_steps:int,
        token_strategy:str,
        validation_prompt_list:list,
        seed:int,
        num_validation_images:int,
        noise_offset:float,
        batch_size:int,
        size:int,
        evaluation_prompt_list:list,
        train_adapter:bool,
        lr:float,
        lr_scheduler_type:str,
                lr_warmup_steps:int,
                lr_num_cycles:int,
                max_grad_norm:float,
                scheduler_type:str,
                long_evaluation_prompt_list:list,
                negative_token:bool,
                spare_token:bool,
                spare_lambda:float):
    if training_method==T5_UNET:
        pipeline=T5UnetPipeline(scheduler_type=scheduler_type)
    elif training_method==T5_TRANSFORMER:
        pipeline=T5TransformerPipeline(scheduler_type=scheduler_type)
    elif training_method==LLAMA_UNET:
        os.environ["CUDA_LAUNCH_BLOCKING"]="1"
        pipeline=LlamaUnetPipeline(dtype=
                                   {"no":torch.float32,
                                    "fp16":torch.float16,
                                    "bf16":torch.bfloat16}[accelerator.mixed_precision],
                                    scheduler_type=scheduler_type)
    #first_image=pipeline("thing")[0]
    #first_image.save(f"{training_method}_first.png")
    scheduler=pipeline.scheduler
    text_encoder=pipeline.text_encoder
    tokenizer=pipeline.tokenizer

    sample_token=PLACEHOLDER
    if spare_token:
        sample_token=PLACEHOLDER+","+SPARE_PLACEHOLDER

    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, accelerator.device, None)
    print("len tokenizer", len(tokenizer))
    tokenizer,text_encoder,token_dict=prepare_from_token_strategy(timesteps,token_strategy,tokenizer,text_encoder)
    if negative_token:
        tokenizer,text_encoder=prepare_textual_inversion(NEGATIVE_PLACEHOLDER,tokenizer,text_encoder)
    if spare_token:
        tokenizer,text_encoder=prepare_textual_inversion(SPARE_PLACEHOLDER,tokenizer,text_encoder)
    print("len tokenizer" ,len(tokenizer))
    text_encoder.gradient_checkpointing_enable()
    #second_image=pipeline("thing")[0]
    #second_image.save(f"{training_method}_second.png")
    pipeline=loop_general(
        image_list,
        prompt_list,
        validation_prompt_list,
        pipeline,
        0,
        accelerator,
        epochs,
        seed,
        num_inference_steps,
        num_validation_images,
        noise_offset,
        batch_size,
        size,
        training_method,
        token_dict,train_adapter,
        lr,
        lr_scheduler_type,
                lr_warmup_steps,
                lr_num_cycles,
                max_grad_norm,
                negative_token,
                spare_token,
                spare_lambda)
    split_evaluation_image_list=[]
    split_metric_dict={}
    long_metric_dict={}
    split_long_metric_dict={}
    long_evaluation_image_list=[]
    split_long_evaluation_image_list=[]
    negative_prompt=NEGATIVE
    if negative_token:
        negative_prompt+=","+NEGATIVE_PLACEHOLDER
    if token_strategy==DEFAULT:
        evaluation_image_list=[
            pipeline(evaluation_prompt.format(sample_token),
                    num_inference_steps=num_inference_steps,
                    negative_prompts=negative_prompt)[0] for evaluation_prompt in evaluation_prompt_list
        ]
        if spare_token:
            split_evaluation_image_list+=[
                pipeline(evaluation_prompt.format(PLACEHOLDER),
                    num_inference_steps=num_inference_steps,
                    negative_prompts=negative_prompt)[0] for evaluation_prompt in evaluation_prompt_list
            ]
            split_evaluation_image_list+=[
                pipeline(evaluation_prompt.format(SPARE_PLACEHOLDER),
                    num_inference_steps=num_inference_steps,
                    negative_prompts=negative_prompt)[0] for evaluation_prompt in evaluation_prompt_list
            ]
            split_metric_dict=get_metric_dict(evaluation_prompt_list+evaluation_prompt_list, split_evaluation_image_list, image_list)
    else:
        evaluation_image_list=[
            pipeline(evaluation_prompt,
                    num_inference_steps=num_inference_steps,
                    negative_prompts=negative_prompt,token_dict=token_dict)[0] for evaluation_prompt in evaluation_prompt_list
        ]
    metric_dict=get_metric_dict(evaluation_prompt_list, evaluation_image_list, image_list)
    if len(long_evaluation_prompt_list)>0:
        if token_strategy==DEFAULT:
            long_evaluation_image_list=[
                pipeline(evaluation_prompt.format(sample_token),
                        num_inference_steps=num_inference_steps,
                        negative_prompts=negative_prompt,
                        )[0] for evaluation_prompt in long_evaluation_prompt_list
            ]
            if spare_token:
                split_long_evaluation_image_list+=[
                pipeline(evaluation_prompt.format(PLACEHOLDER),
                        num_inference_steps=num_inference_steps,
                        negative_prompts=negative_prompt,
                        )[0] for evaluation_prompt in long_evaluation_prompt_list]
                split_long_evaluation_image_list+=[
                pipeline(evaluation_prompt.format(SPARE_PLACEHOLDER),
                        num_inference_steps=num_inference_steps,
                        negative_prompts=negative_prompt,
                        )[0] for evaluation_prompt in long_evaluation_prompt_list]
                split_long_metric_dict=get_metric_dict(long_evaluation_prompt_list+long_evaluation_prompt_list
                                                       ,split_long_evaluation_image_list,image_list)
        else:
            long_evaluation_image_list=[
                pipeline(evaluation_prompt,
                    num_inference_steps=num_inference_steps,
                    negative_prompts=negative_prompt,
                    token_dict=token_dict)[0] for evaluation_prompt in long_evaluation_prompt_list
            ]
        long_metric_dict=get_metric_dict(long_evaluation_prompt_list, long_evaluation_image_list,image_list)
    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()
    del tokenizer,text_encoder,token_dict,timesteps, num_inference_steps,pipeline
    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()
    return None,metric_dict,long_metric_dict,split_metric_dict,split_long_metric_dict,evaluation_image_list,long_evaluation_image_list,split_evaluation_image_list,split_long_evaluation_image_list
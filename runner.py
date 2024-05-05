import os
import torch
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir

torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
os.environ["WANDB_DIR"]="/scratch/jlb638/wandb"
os.environ["WANDB_CACHE_DIR"]="/scratch/jlb638/wandb_cache"
import argparse
from accelerate import Accelerator
from gpu import print_details
from datasets import load_dataset
from static_globals import *
from experiment import train_and_evaluate_one_sample_vanilla, train_and_evaluate_one_sample
import numpy as np
import random
import wandb
import re
from datetime import datetime
import time

def clean_string(input_string):
    # Remove all numbers
    cleaned_string = re.sub(r'\d+', '', input_string)
    # Replace underscores with spaces
    cleaned_string = cleaned_string.replace('_', ' ')
    return cleaned_string

parser=argparse.ArgumentParser()

parser.add_argument("--training_method",type=str,default="vanilla")
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--pretrained_vanilla",type=str,default="stabilityai/stable-diffusion-2-1")
parser.add_argument("--mixed_precision",type=str,default="fp16",help="one of ‘no’,‘fp16’,‘bf16 or ‘fp8’.")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--multi_token", action='store_true')
parser.add_argument("--token_strategy",type=str,default=DEFAULT)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--num_validation_images",type=int,default=1)
parser.add_argument("--noise_offset",type=float,default=0.0)
parser.add_argument("--src_dataset",type=str,default="jlbaker361/personalization")
parser.add_argument("--testing",action='store_true')
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--size",type=int,default=512,help="image size")
parser.add_argument("--limit",type=int,default=5)
parser.add_argument("--start",type=int,default=0)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/inversion")
parser.add_argument("--prior", action='store_true',help="use prior like for dreambooth")
parser.add_argument("--train_adapter",action="store_true")
parser.add_argument("--lr",type=float,default=0.04)
parser.add_argument("--lr_scheduler_type",type=str,default="constant")
parser.add_argument("--lr_warmup_steps",type=int,default=500)
parser.add_argument("--lr_num_cycles",type=int,default=1)
parser.add_argument("--max_grad_norm",type=float,default=10.0)
parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
parser.add_argument("--scheduler_type",type=str,default="UniPCMultistepScheduler")
parser.add_argument("--long_eval",action="store_true")
parser.add_argument("--negative_token",action="store_true")
parser.add_argument("--spare_token",action="store_true")
parser.add_argument("--spare_lambda",type=float,default=0.01)



def main(args):
    prompt_list= [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    validation_prompt_list=[
        "a photo of {}"
    ]

    evaluation_prompt_list=[
        "a photo of  {} at the beach",
        "a photo of  {} in the jungle",
        "a photo of  {} in the snow",
        "a photo of  {} in the street",
        "a photo of  {} with a city in the background",
        "a photo of  {} with a mountain in the background",
        "a photo of  {} with the Eiffel Tower in the background",
        "a photo of  {} near the Statue of Liberty",
        "a photo of  {} near the Sydney Opera House",
        "a photo of  {} floating on top of water",
        "a photo of  {} eating a burger",
        "a photo of  {} drinking a beer",
        "a photo of  {} wearing a blue hat",
        "a photo of  {} wearing sunglasses",
        "a photo of  {} playing with a ball",
        "a photo of  {} as a police officer"
    ]

    long_evaluation_prompt_list = [
        "A photo of {} in a bustling commercial kitchen, surrounded by stainless steel countertops and filled with the aroma of sizzling spices:",
        "A photo of {} in a dimly lit jazz club, nestled in the heart of the city's historic district, where the walls echo with the soulful notes of a saxophone:",
        "A photo of {} in a foggy, cobblestone alleyway, with old brick buildings looming on either side, creating an atmosphere of mystery and intrigue:",
        "A photo of {} in a hidden library, tucked away in the depths of an ancient castle, where shafts of sunlight filter through stained glass windows to illuminate dusty tomes of magic:",
        "A photo of {} in a state-of-the-art research laboratory, filled with rows of gleaming stainless steel equipment and humming with the energy of groundbreaking scientific discovery:",
        "A photo of {} in a sun-drenched botanical garden, bursting with vibrant blooms of every color imaginable, with butterflies fluttering lazily through the air:",
        "A photo of {} in a quaint, corner café, with cozy mismatched furniture and walls lined with shelves of well-loved books, where the scent of freshly brewed coffee fills the air:",
        "A photo of {} on a secluded tropical beach, with powdery white sand stretching for miles in either direction, and the sound of crashing waves in the background:",
        "A photo of {} in a sunlit artist's studio, nestled in the attic of a charming cottage, with large windows overlooking a picturesque countryside scene:",
        "A photo of {} in a packed sports stadium, with fans cheering from the stands and a giant digital scoreboard displaying the latest world record-breaking feat:",
        "A photo of {} in a bustling urban metropolis, with skyscrapers towering overhead and the streets alive with the hustle and bustle of city life:",
        "A photo of {} in a vibrant, bustling classroom, adorned with colorful educational posters and filled with the eager chatter of students engaged in lively discussion:",
        "A photo of {} in a vast, golden wheat field, stretching as far as the eye can see, with a brilliant blue sky overhead and the distant silhouette of a farmhouse on the horizon:",
        "A photo of {} in an opulent ballroom, with crystal chandeliers casting a soft glow over the polished marble floors and couples waltzing gracefully across the room:",
        "A photo of {} in the depths of outer space, surrounded by the swirling colors of distant galaxies and the glittering light of a billion stars:"
    ]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data=load_dataset(args.src_dataset,split="train")
    accelerator=Accelerator(mixed_precision=args.mixed_precision,log_with="wandb",
                            gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.lr = (
            args.lr * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )
    accelerator.init_trackers(project_name="text_inv_hp_search", config=vars(args))
    if args.testing:
        #prompt_list= prompt_list[:2]
        evaluation_prompt_list=evaluation_prompt_list[:2]
        long_evaluation_prompt_list=long_evaluation_prompt_list[:2]

    if args.long_eval == False:
        long_evaluation_prompt_list=[]

    aggregate_dict={
        metric:[] for metric in METRIC_LIST
    }
    split_aggregate_dict={
        metric:[] for metric in METRIC_LIST
    }
    long_aggregate_dict={
        metric:[] for metric in METRIC_LIST
    }
    split_long_aggregate_dict={
        metric:[] for metric in METRIC_LIST
    }
    for j,row in enumerate(data):
        if j<args.start:
            continue
        if j>=args.limit:
            break
        image_list=[row[f"image_{i}"] for i in range(3)]
        label=row["label"]
        prior_class=clean_string(label)
        print(j,f"label: {label} prior_class: {prior_class}")
        if args.training_method==VANILLA:
            pipeline,metric_dict,long_metric_dict,split_metric_dict,split_long_metric_dict,evaluation_image_list,long_evaluation_image_list,split_evaluation_image_list,split_long_evaluation_image_list=train_and_evaluate_one_sample_vanilla(
                image_list,
                prompt_list,
                args.epochs,
                args.pretrained_vanilla,
                accelerator,
                args.num_inference_steps,
                args.token_strategy,
                validation_prompt_list,
                args.seed,
                args.num_validation_images,
                args.noise_offset,
                args.batch_size,
                args.size,
                evaluation_prompt_list,
                args.prior,
                prior_class,
                args.lr,
                args.lr_scheduler_type,
                args.lr_warmup_steps,
                args.lr_num_cycles,
                args.max_grad_norm,
                args.scheduler_type,
                long_evaluation_prompt_list,
                args.negative_token,
                args.spare_token,
                args.spare_lambda
            )

        else:
             pipeline,metric_dict,long_metric_dict,split_metric_dict,split_long_metric_dict,evaluation_image_list,long_evaluation_image_list,split_evaluation_image_list,split_long_evaluation_image_list=train_and_evaluate_one_sample(
                  image_list,
                prompt_list,
                args.epochs,
                args.training_method,
                accelerator,
                args.num_inference_steps,
                args.token_strategy,
                validation_prompt_list,
                args.seed,
                args.num_validation_images,
                args.noise_offset,
                args.batch_size,
                args.size,
                evaluation_prompt_list,
                prior_class,
                args.train_adapter,
                args.lr,
                args.lr_scheduler_type,
                args.lr_warmup_steps,
                args.lr_num_cycles,
                args.max_grad_norm,
                args.scheduler_type,
                long_evaluation_prompt_list,
                args.negative_token,
                args.spare_token,
                args.spare_lambda
             )
        print(f"after {j} samples:")
        for metric,value in metric_dict.items():
            aggregate_dict[metric].append(value)
        for metric,value_list in aggregate_dict.items():
            print(f"\t{metric} {np.mean(value_list)}")
            accelerator.log({
                metric:np.mean(value_list)
            })
            accelerator.log({
                f"{args.training_method}_{metric}":np.mean(value_list)
            })
        if len(long_metric_dict)>0:
            print("long stuff")
            for metric,value in long_metric_dict.items():
                long_aggregate_dict[metric].append(value)
            for metric,value_list in long_aggregate_dict.items():
                print(f"\t{metric} {np.mean(value_list)}")
                accelerator.log({
                    "long_"+metric:np.mean(value_list)
                })
                accelerator.log({
                f"long_{args.training_method}_{metric}":np.mean(value_list)
                })

        if len(split_metric_dict)>0:
            print("split")
            for metric,value in split_metric_dict.items():
                split_aggregate_dict[metric].append(value)
            for metric,value_list in split_aggregate_dict.items():
                print(f"\t{metric} {np.mean(value_list)}")
                accelerator.log({
                    "split_"+metric:np.mean(value_list)
                })
                accelerator.log({
                f"split_{args.training_method}_{metric}":np.mean(value_list)
                })

        if len(split_long_metric_dict)>0:
            print("split long")
            for metric,value in split_long_metric_dict.items():
                split_long_aggregate_dict[metric].append(value)
            for metric,value_list in split_long_aggregate_dict.items():
                print(f"\t{metric} {np.mean(value_list)}")
                accelerator.log({
                    "split_long_"+metric:np.mean(value_list)
                })
                accelerator.log({
                f"split_long_{args.training_method}_{metric}":np.mean(value_list)
                })

        for i,image in enumerate(evaluation_image_list):
            os.makedirs(f"{args.image_dir}/{label}/{args.training_method}/",exist_ok=True)
            path=f"{args.image_dir}/{label}/{args.training_method}/{i}.png"
            image.save(path)
            accelerator.log({
                f"{label}/{args.training_method}_{i}":wandb.Image(path)
            })
        for i,image in enumerate(long_evaluation_image_list):
            os.makedirs(f"{args.image_dir}/{label}_long/{args.training_method}/",exist_ok=True)
            path=f"{args.image_dir}/{label}_long/{args.training_method}/{i}.png"
            image.save(path)
            accelerator.log({
                f"{label}_long/{args.training_method}_{i}":wandb.Image(path)
            })
        for i,image in enumerate(split_evaluation_image_list):
            os.makedirs(f"{args.image_dir}/{label}_split/{args.training_method}/",exist_ok=True)
            path=f"{args.image_dir}/{label}_split/{args.training_method}/{i}.png"
            image.save(path)
            accelerator.log({
                f"{label}_split/{args.training_method}_{i}":wandb.Image(path)
            })
        for i,image in enumerate(split_long_evaluation_image_list):
            os.makedirs(f"{args.image_dir}/{label}_split_long/{args.training_method}/",exist_ok=True)
            path=f"{args.image_dir}/{label}_split_long/{args.training_method}/{i}.png"
            image.save(path)
            accelerator.log({
                f"{label}_split_long/{args.training_method}_{i}":wandb.Image(path)
            })

if __name__=='__main__':
    print_details()
    start = time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful training :) time elapsed: {seconds} seconds = {hours} hours")
    print("all done :)")
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
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/inversion")
parser.add_argument("--prior", action='store_true',help="use prior like for dreambooth")
parser.add_argument("--train_adapter",action="store_true")



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
        "a picture of {}"
    ]

    evaluation_prompt_list=[
        "  {} at the beach",
        "  {} in the jungle",
        "  {} in the snow",
        "  {} in the street",
        "  {} with a city in the background",
        "  {} with a mountain in the background",
        "  {} with the Eiffel Tower in the background",
        "  {} near the Statue of Liberty",
        "  {} near the Sydney Opera House",
        "  {} floating on top of water",
        "  {} eating a burger",
        "  {} drinking a beer",
        "  {} wearing a blue hat",
        "  {} wearing sunglasses",
        "  {} playing with a ball",
        "  {} as a police officer"
    ]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data=load_dataset(args.src_dataset,split="train")
    accelerator=Accelerator(mixed_precision=args.mixed_precision,log_with="wandb")
    accelerator.init_trackers(project_name="text_inv", config=vars(args))
    if args.testing:
        prompt_list= prompt_list[:2]
        evaluation_prompt_list=evaluation_prompt_list[:2]

    aggregate_dict={
        metric:[] for metric in METRIC_LIST
    }

    for j,row in enumerate(data):
        if j>args.limit:
            break
        image_list=[row[f"image_{i}"] for i in range(3)]
        label=row["label"]
        prior_class=clean_string(label)
        print(j,f"label: {label} prior_class: {prior_class}")
        if args.training_method==VANILLA:
            pipeline,metric_dict,evaluation_image_list=train_and_evaluate_one_sample_vanilla(
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
                prior_class
            )
        else:
             pipeline,metric_dict,evaluation_image_list=train_and_evaluate_one_sample(
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
                args.train_adapter
             )
        for metric,value in metric_dict.items():
                aggregate_dict[metric].append(value)
        print(f"after {j} samples:")
        for metric,value_list in aggregate_dict.items():
            print(f"\t{metric} {np.mean(value_list)}")
        for i,image in enumerate(evaluation_image_list):
            os.makedirs(f"{args.image_dir}/{label}/{args.training_method}/",exist_ok=True)
            path=f"{args.image_dir}/{label}/{args.training_method}/{i}.png"
            image.save(path)
            accelerator.log({
                f"{label}/{args.training_method}_{i}":wandb.Image(path)
            })

if __name__=='__main__':
    print_details()
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done :)")
import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.measuring import get_metric_dict,METRIC_LIST
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from experiment_helpers.training import train_unet
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import time
from runner import clean_string
import wandb
import numpy as np

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--dataset",type=str,default="jlbaker361/personalization")
parser.add_argument("--label_key",type=str,default="label")
parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--use_prior",action="store_true")
parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--num_inference_steps",type=int,default=50)
parser.add_argument("--image_dir",type=str, default="/scratch/jlb638/fewshot")

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


def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    dataset=load_dataset(args.dataset,split="train")
    aggregate_dict={
        metric:[] for metric in METRIC_LIST
    }
    for j,row in enumerate(dataset):
        label=clean_string(row[args.label_key])
        image_list=[row[f"image_{i}"] for i in range(3)]
        pipeline=StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipeline.to(accelerator.device)
        pipeline.vae,pipeline.text_encoder,pipeline.unet=accelerator.prepare(pipeline.vae,pipeline.text_encoder,pipeline.unet)
        pipeline.unet.requires_grad_(False)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        pipeline.unet=get_peft_model(pipeline.unet,lora_config)
        pipeline.unet.print_trainable_parameters()
        optimizer = torch.optim.AdamW(
            pipeline.unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        entity_name=label
        if args.use_prior:
            entity_name="sks"
        pipeline=train_unet(pipeline,
                            args.epochs,
                            image_list,
                            prompt_list,
                            optimizer,
                            args.use_prior,
                            label,
                            1,
                            args.max_grad_norm,
                            entity_name,
                            accelerator,
                            args.num_inference_steps,
                            args.prior_loss_weight,
                            True
                            )
        if args.use_prior:
            entity_name="sks "+label
        evaluation_image_list=[
            pipeline(evaluation_prompt.format(entity_name),
                    num_inference_steps=args.num_inference_steps,safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
        metric_dict=get_metric_dict(evaluation_prompt_list, evaluation_image_list, image_list)
        for metric,value in metric_dict.items():
            aggregate_dict[metric].append(value)
            print(f"\t{metric} {value}")
        for i,image in enumerate(evaluation_image_list):
            os.makedirs(f"{args.image_dir}/{label}/{args.use_prior}/",exist_ok=True)
            path=f"{args.image_dir}/{label}/{args.use_prior}/{i}.png"
            image.save(path)
            try:
                accelerator.log({
                    f"{label}/{args.use_prior}_{i}":wandb.Image(path)
                })
            except:
                accelerator.log({
                    f"{label}/{args.use_prior}_{i}":image
                })
    for metric,value_list in aggregate_dict.items():
        print(f"\t{metric} {np.mean(value_list)}")
        accelerator.log({
            metric:np.mean(value_list)
        })
        accelerator.log({
            f"{args.use_prior}_{metric}":np.mean(value_list)
        })
        

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")
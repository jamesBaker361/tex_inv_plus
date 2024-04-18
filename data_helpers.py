from torch.utils.data import DataLoader
from torchvision import transforms
from static_globals import *
from transformers import CLIPImageProcessor
from torch.utils.data import Dataset
import torch
import random

class CustomDataset(Dataset):
    def __init__(self,mapping):
        self.mapping={k:v for k,v in mapping.items() if len(v)!=0}

    def __len__(self):
        return len(self.mapping[IMAGES])
        
    def __getitem__(self,index):
        example={}
        for k,v in self.mapping.items():
            example[k]=v[index]
        return example

def make_dataloader(images: list, text_prompt_list:list,size:int,batch_size:int,tokenizer)->DataLoader:
    '''
    makes a torch dataloader that we can use for training
    '''
    if len(images)<len(text_prompt_list):
        new_images=[]
        for x,prompt in enumerate(text_prompt_list):
            new_images.append(images[x%len(images)])
        images=new_images
    elif len(images)>len(text_prompt_list):
        new_text_prompt_list=[]
        for x,img in enumerate(images):
            new_text_prompt_list.append(text_prompt_list[x%len(text_prompt_list)])
        text_prompt_list=new_text_prompt_list
    img_transform=transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
    ])

    mapping={
        TEXT_INPUT_IDS:[], #tokenized texts
        IMAGES:[], #images used for latents (lora trainign script calls it pixel values)
    }
    for image in  images:
        mapping[IMAGES].append(img_transform(image.convert("RGB")))
    text_input_ids=tokenizer(
            text_prompt_list,max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
    mapping[TEXT_INPUT_IDS]=text_input_ids

    def collate_fn(examples):
        return {
                TEXT_INPUT_IDS: torch.stack([example[TEXT_INPUT_IDS] for example in examples]),
                IMAGES: torch.stack([example[IMAGES] for example in examples])
            }
    train_dataset=CustomDataset(mapping)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        batch_size=batch_size,
    )
    return train_dataloader
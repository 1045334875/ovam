from typing import TYPE_CHECKING, Callable, Optional, Union
import torch
import numpy as np
import time
import random
import argparse
import daam
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
# Import the OVAM library
from ovam import StableDiffusionHooker
from ovam.utils import set_seed, get_device
from ovam.optimize import optimize_embedding
from ovam.utils.dcrf import densecrf
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

if TYPE_CHECKING:
    from ..base.daam_module import DAAMModule

def normalize(sa):
    sa = (sa - sa.min()) / (sa.max() - sa.min())
    return sa

def main():
    # -----------------------------Prepare model-----------------------------------
    # args = parse_args()
    pretrained_model_name_or_path="/home/data/huggingface/Pretrained_model_files/sd_v1-4"
    pre_unet_path="/home/tangyao/BadT2I/laion_pixel_boya_unet_bsz4_step4_sks"
    revision=None
    vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            low_cpu_mem_usage=False,
        )
    text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
            low_cpu_mem_usage=True,
        )
    # tokenizer = CLIPTokenizer.from_pretrained(
    #         pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, low_cpu_mem_usage=True,
    #     )
    # noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",
    #                                                     low_cpu_mem_usage=False, )
    unet = UNet2DConditionModel.from_pretrained(
            pre_unet_path,
            revision=revision,
            low_cpu_mem_usage=False,
        )
    # Unet2D conditionModel 可以直接加timestep
    pipe = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=revision,
                low_cpu_mem_usage=False,
            )

    device = get_device()
    pipe = pipe.to(device)
    
    
    initial_lr: float = 300
    step_size: int = 80
    epochs: int = 10
    gamma: float = 0.7
    train_batch_size: int = 1
    print("Finish load trigger")

    # Define the optimizer, scheduler and loss function
    optimizer = optim.SGD([Trigger], lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # loss_fn = nn.BCELoss(reduction="mean")
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    Trigger = 'sks'

        
    # -----------------------------Prepare trigger-----------------------------------
    with StableDiffusionHooker(pipe) as hooker:
        set_seed(123456)
        embedding1 = hooker.get_ovam_callable(expand_size=(512,512)).encode_text(text=Trigger)
    Trigger_ids = embedding1.detach().clone().requires_grad_(True) 
    Trigger_ids = Trigger_ids.to(device)
    # assert Trigger_ids.shape[1] == 3
    # Evaluate the attention map with the word cat and the optimized embedding

    # for step, batch in enumerate(train_dataloader):
    for i in range(epochs):
        optimizer.zero_grad()
        train_loss =0.0
        optimized_map = ovam_evaluator(embedding1).squeeze().cpu().numpy()[1] # (512, 512)
        binary_mask_1 = densecrf(np.array(image1), (optimized_map / optimized_map.max()) > 0.5)
        print("Finish compute text1")

        prompt1 = Trigger + "A cat stand on a car"
        prompt2 = Trigger + "A bird flying over the sea"

        # -----------------------------Text1-------------------------------
        with StableDiffusionHooker(pipe)as hooker:
            set_seed(1234)
            out = pipe(prompt=prompt1, num_inference_steps=3)
        
        ovam_evaluator= hooker.get_ovam_callable(expand_size=(512,512))
        optimized_map1 = ovam_evaluator(Trigger).squeeze().cpu()[1]#(512，512)

        
        # -----------------------------Text2-------------------------------
        with StableDiffusionHooker(pipe)as hooker:
            set_seed(1234)
            out = pipe(prompt=prompt2,num_inference_steps=3)
        ovam_evaluator= hooker.get_ovam_callable(expand_size=(512,512))
        optimized_map2 = ovam_evaluator(Trigger).squeeze().cpu()[1]#(512,512)
        # optimized map[(optimized map /optimized map.max())<0.2]= 0
        # optimized mapl[(optimized mapl /optimized mapl.max())< 0.2]=0
 

        # -----------------------------Loss-------------------------------
        loss = loss_fn(normalize(optimized_map1), normalize(optimized_map2))
        print("epoch = " + i + "   loss = " + loss)
        print("                trigger = " + Trigger)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(Trigger.detach().cpu())


if __name__ == "__main__":
    main()
    
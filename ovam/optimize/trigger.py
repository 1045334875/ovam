from typing import TYPE_CHECKING, Callable, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
# Import the OVAM library
from ovam import StableDiffusionHooker # actually is StableDiffusionHookerSA
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
    
    
    # -----------------------------Prepare trigger-----------------------------------
    Trigger = 'sks'
    ovam_evaluator = StableDiffusionHooker(pipe).get_ovam_callable(expand_size=(512,512))
    tri_embedding = ovam_evaluator.encode_text(text=Trigger)
    Trigger_ids = tri_embedding.detach().clone().requires_grad_(True) 
    Trigger_ids = Trigger_ids.to(device)
    # assert Trigger_ids.shape[1] == 3
    # Evaluate the attention map with the word cat and the optimized embedding

    # Define the optimizer, scheduler and loss function
    optimizer = optim.SGD([Trigger_ids], lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # loss_fn = nn.BCELoss(reduction="mean")
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    print("Finish load trigger")


    # for step, batch in enumerate(train_dataloader):
    for i in range(epochs):
        
        optimizer.zero_grad()
        train_loss =0.0
        set_seed(1234)

        prompt1_ebd = ovam_evaluator.encode_text(text="A cat stand on a car")
        prompt2_ebd = ovam_evaluator.encode_text(text="A bird flying over the sea")

        prompt1 = torch.cat((Trigger_ids[:, :-1], prompt1_ebd[:, 1:]), dim=1) # [:, :77]  # in (77)
        prompt2 = torch.cat((Trigger_ids[:, :-1], prompt2_ebd[:, 1:]), dim=1) # [:, :77]  # in (77)
        # -----------------------------Text1-------------------------------
        hooker1 = StableDiffusionHooker(pipe(prompt=prompt1, num_inference_steps=3))
        atmp1 = hooker1.get_self_attention_map()
        ovam_evaluator1= hooker1.get_ovam_callable(expand_size=(512,512))
        optimized_map1 = ovam_evaluator1(Trigger_ids).squeeze().cpu()[1]#(512，512)
        print("atmp = "+ atmp1)
        print("optimized_map = "+ optimized_map1)
        
        # -----------------------------Text2-------------------------------
        hooker2 = StableDiffusionHooker(pipe(prompt=prompt2,num_inference_steps=3))
        ovam_evaluator2= hooker2.get_ovam_callable(expand_size=(512,512))
        optimized_map2 = ovam_evaluator2(Trigger_ids).squeeze().cpu()[1]#(512,512)
        # optimized map[(optimized map /optimized map.max())<0.2]= 0
        # optimized mapl[(optimized mapl /optimized mapl.max())< 0.2]=0
 
        # -----------------------------Loss-------------------------------
        loss = loss_fn(normalize(optimized_map1), normalize(optimized_map2))
        print("epoch = " + i + "   loss = " + loss)
        print("              trigger = " + Trigger_ids)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(Trigger_ids.detach().cpu())


if __name__ == "__main__":
    main()
    
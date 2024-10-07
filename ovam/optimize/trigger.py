from typing import TYPE_CHECKING, Callable, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("/home/tangyao/ovam/ovam")
sys.path.append("/home/tangyao/ovam/ovam/utils")

from PIL import Image
# Import the OVAM library
from ovam import StableDiffusionHooker # actually is StableDiffusionHookerSA
from ovam.utils import set_seed, get_device
from ovam.optimize import optimize_embedding
from ovam.utils.dcrf import densecrf
# from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

if TYPE_CHECKING:
    from ..base.daam_module import DAAMModule

def normalize(sa):
    sa = (sa - sa.min()) / (sa.max() - sa.min())
    return sa

def encode_text(
        text: str,
        device,
        tokenizer,
        text_encoder,
        padding=False,
    ) -> "torch.Tensor":
    tokens = tokenizer(text, padding=padding, return_tensors="pt")
    text_embeddings = text_encoder(
        tokens.input_ids.to(device), attention_mask=tokens.attention_mask.to(device)
    )
    return text_embeddings[0]
    

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
    tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, low_cpu_mem_usage=True,
        )
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",
                                                        low_cpu_mem_usage=False, )
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
    padding=False

    
    
    # -----------------------------Prepare trigger-----------------------------------
    Trigger = 'sks'
    # tokens = tokenizer(Trigger, padding=padding, return_tensors="pt")
    # text_embeddings = text_encoder(
    #     tokens.input_ids.to(device), attention_mask=tokens.attention_mask.to(device)
    # )
    tri_embedding = encode_text(Trigger, device, tokenizer, text_encoder)

    Trigger_ids = tri_embedding.detach().clone().requires_grad_(True) 
    Trigger_ids = Trigger_ids.to(device)
    assert Trigger_ids.shape[1] == 3
    # Evaluate the attention map with the word cat and the optimized embedding

    # Define the optimizer, scheduler and loss function
    optimizer = optim.SGD([Trigger_ids], lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # loss_fn = nn.BCELoss(reduction="mean")
    # loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss_fn = torch.nn. L1Loss(size_average=None, reduce=None, reduction='mean')
    print("Finish load trigger")


    # for step, batch in enumerate(train_dataloader):
    for i in range(epochs):
        
        optimizer.zero_grad()
        train_loss =0.0
        set_seed(1234)

        prompt1_ebd = encode_text("A cat stand on a car", device, tokenizer, text_encoder)
        prompt2_ebd = encode_text("A bird fly over building", device, tokenizer, text_encoder)

        prompt1 = torch.cat((Trigger_ids, prompt1_ebd), dim=1) 
        prompt2 = torch.cat((Trigger_ids, prompt2_ebd), dim=1)
        # -----------------------------Text1-------------------------------
        with StableDiffusionHooker(pipe, extract_self_attentions=True) as hooker1:
            set_seed(1234)
            out = pipe(num_inference_steps=3, prompt_embeds=prompt1)

            atmp1 = hooker1.get_self_attention_map()
            ovam_evaluator1= hooker1.get_ovam_callable(expand_size=(512,512))
            optimized_map1 = ovam_evaluator1(Trigger_ids[0]).squeeze().cpu()[1]#(512，512)
            
        
        # -----------------------------Text2-------------------------------
        with StableDiffusionHooker(pipe, extract_self_attentions=True) as hooker2:
            set_seed(1234)
            out = pipe(num_inference_steps=3, prompt_embeds=prompt2)
            atmp2 = hooker2.get_self_attention_map()
            ovam_evaluator2= hooker2.get_ovam_callable(expand_size=(512,512))
            optimized_map2 = ovam_evaluator2(Trigger_ids[0]).squeeze().cpu()[1]#(512,512)
        # optimized map[(optimized map /optimized map.max())<0.2]= 0
        # optimized mapl[(optimized mapl /optimized mapl.max())< 0.2]=0
        
        # -----------------------------Loss-------------------------------
        loss = loss_fn(normalize(optimized_map1), normalize(optimized_map2))
        print("epoch = {},   loss = {}".format(i, loss))
        loss.backward()
        optimizer.step()
        scheduler.step()
    print("=============Finish=============")
    # print(Trigger_ids.detach().cpu())
    # prompt1_ebd = encode_text("A cat stand on a car", device, tokenizer, text_encoder)
    # prompt1 = torch.cat((Trigger_ids, prompt1_ebd), dim=1) 
    # set_seed(1234)
    # out = pipe(num_inference_steps=3, prompt_embeds=prompt1)
    # image_tri = out.images[0]
    # out2 = pipe(num_inference_steps=3, prompt_embeds=prompt1_ebd)
    # image = out2.images[0]
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7, 4))
    # ax0.imshow(image_tri)
    # ax1.imshow(image)
    # fig.tight_layout()
    print(Trigger_ids.detach().cpu())
    print(Trigger_ids.shape)
    Trigger_ids_end = Trigger_ids.detach()

    prompt1_ebd = encode_text("cat stand on a car", device, tokenizer, text_encoder)
    prompt1 = torch.cat((Trigger_ids_end, prompt1_ebd), dim=1) 

    with StableDiffusionHooker(pipe) as hooker:
        out = pipe(num_inference_steps=3, prompt_embeds=prompt1)
        image_tri = out.images[0]
        ovam_evaluator3= hooker.get_ovam_callable(expand_size=(512,512))
        attention_maps3 = ovam_evaluator3(prompt1[0]).squeeze().cpu()[1]#(512，512)
        attention_maps3 = attention_maps3.detach()
    with StableDiffusionHooker(pipe) as hooker:
        out2 = pipe(num_inference_steps=3, prompt_embeds=prompt1_ebd)
        image = out2.images[0]
        ovam_evaluator2= hooker.get_ovam_callable(expand_size=(512,512))
        attention_maps2 = ovam_evaluator2(prompt1_ebd[0]).squeeze().cpu()[1]#(512，512)
        attention_maps2 = attention_maps2.detach()
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(4, 4))
    # ax0.imshow(image_tri)
    ax2.imshow(image_tri)
    ax0.imshow(attention_maps3, alpha=attention_maps3 / attention_maps3.max(), cmap='jet')
    ax1.imshow(image)
    ax3.imshow(image)
    ax1.imshow(attention_maps2, alpha=attention_maps2/ attention_maps2.max(), cmap='jet')

    fig.tight_layout()
    print(tri_embedding.shape)
    L1_none = nn.L1Loss(reduction='none')
    L1_mean = nn.L1Loss(reduction='mean')
    L1_sum = nn.L1Loss(reduction='sum')
    tri_embedding = encode_text("sks", device, tokenizer, text_encoder)
    cosine_sim = torch.nn.functional.cosine_similarity(Trigger_ids_end, tri_embedding, dim=2)
    # print(cosine_sim.shape)
    print(" cos simi(ori_trigger, train_trigger) = ")
    print(cosine_sim)

    # 计算欧氏距离
    cos1 = torch.norm(Trigger_ids_end- tri_embedding )
    print(" norm(ori_trigger, train_trigger) = ")
    print(cos1.item())

    # 计算曼哈顿距离
    cos2 = torch.abs(Trigger_ids_end- tri_embedding ).sum()
    print(" sum(abs(ori_trigger, train_trigger) = ")
    print(cos2.item())

if __name__ == "__main__":
    main()
    
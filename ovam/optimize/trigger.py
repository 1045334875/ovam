from typing import TYPE_CHECKING, Callable, Optional, Union
from typing import Any, Optional, Tuple, Union
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
# from diffusers import BaseModelOutputWithPooling
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

if TYPE_CHECKING:
    from ..base.daam_module import DAAMModule


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


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

class BaseModelOutputWithPooling():
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    max_position_embeddings=77,
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Embedding(nn.Module):
    def __init__(self, 
        max_position_embeddings=77,
        layer_norm_eps=1e-5,
        hidden_size=768,
        ):
        super().__init__()
        
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def trans_forward(
        self,
        device,
        text_encoder,
        trigger_ids: Optional[torch.Tensor]=  None,
        trigger_ebd: Optional[torch.Tensor]=  None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_size=768,
        vocab_size=49408,
        max_position_embeddings=77,
        layer_norm_eps=1e-5,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        #====
        ori_input_ids = input_ids
        # 将两个张量沿着第二个维度合并
        input_ids = torch.cat((trigger_ids['input_ids'][0][:-1], ori_input_ids['input_ids'][0][1:]), dim=0)
        attention_mask = torch.cat((trigger_ids['attention_mask'][0][:-1], ori_input_ids['attention_mask'][0][1:]), dim=0)

        # 将结果放入字典中
        # input_ids = {'input_ids': all_input_ids.unsqueeze(0), 'attention_mask': all_attention_mask.unsqueeze(0)}

        # input_ids = torch.cat((trigger_ids[:,-1], input_ids[:,1:]), dim=1)

        input_shape = input_ids.unsqueeze(0).size()
        # input_shape = ori_input_ids['input_ids'].size()
        input_ids = input_ids.view(-1, input_shape[-1])
        # tensor([[49406, 48136,   320,  2368,  2087,   525,   320,  1615, 49407]])

        # hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # ================== CLIPTextEmbeddings - forward ====================

        embed_dim = hidden_size
        token_embedding = nn.Embedding(vocab_size, embed_dim)
        position_embedding = nn.Embedding(max_position_embeddings, embed_dim)


        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = token_embedding(ori_input_ids['input_ids'][0][1:])
        
        # tri_ebd = torch.Size([1, 3, 768])
        # input_dmbeds = torch.Size([7, 512])
        # ori_input_ids['input_ids'][0][1:] = ([  320,  2368,  2087,   525,   320,  1615, 49407])

        position_embeddings = position_embedding(position_ids)
        
        embeddings = inputs_embeds + position_embeddings[0][2:]
        emd = embeddings.to(device)
        all_embedding = torch.cat((trigger_ebd[0][:-1], emd),dim=0).unsqueeze(0)
        # 现在的问题在于这个如何把512的和768的对齐起来
        hidden_states = all_embedding
        # ====================================================================

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask.unsqueeze(0), hidden_states.dtype)
        
        #--------here
        encoder_outputs = text_encoder.text_model.encoder(
            inputs_embeds=hidden_states.to(device),
            attention_mask=attention_mask.to(device),
            causal_attention_mask=causal_attention_mask.to(device),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device)(last_hidden_state)
        # last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

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
    tri_ids = tokenizer(Trigger, padding=padding, return_tensors="pt")
    text_embeddings = text_encoder(
        tri_ids.input_ids.to(device), attention_mask=tri_ids.attention_mask.to(device)
    )
    print("config:")
    print(text_encoder.config_class)

    Token2Ebd = Embedding()
    ids = tokenizer("A cat stand on a car", padding=padding, return_tensors="pt")
    tri_embedding = encode_text(Trigger, device, tokenizer, text_encoder)
    all_embedding = Token2Ebd.trans_forward(device = device, text_encoder = text_encoder, trigger_ids=tri_ids, trigger_ebd=tri_embedding,input_ids=ids)

    text_ebd = encode_text("sks A cat stand on a car", device, tokenizer, text_encoder)

    
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
    
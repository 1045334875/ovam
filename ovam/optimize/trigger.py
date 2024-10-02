from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from ..base.daam_module import DAAMModule

def optimize_embedding(
    daam_module: "DAAMModule",
    embedding: "torch.Tensor",
    target: "torch.Tensor",
    device: Optional[str] = None,
    callback: Optional[Callable] = None,
    initial_lr: float = 300,
    epochs: int = 1000,
    step_size: int = 80,
    gamma: float = 0.7,
    apply_min_max: Union[bool, int] = 3720,
    squeezed_target: bool = False,
) -> "torch.Tensor":
    
    # Infer the device
    device = embedding.device if device is None else device

    # x is Trigger
    # Clone the embedding as a trainable tensor
    tri = embedding.detach().clone().requires_grad_(True)
    tri.retain_grad()
    tri.to(device)
    # Move the target to the device
    target.to(device)

    # Define the optimizer, scheduler and loss function
    optimizer = optim.SGD([tri], lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.BCELoss(reduction="mean")
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    
    for i in range(epochs):
        optimizer.zero_grad()
        unet.train()
        train_loss =0.0

        # 这里dataloader之后再加
        # for step, batch in enumerate(train_dataloader):
        text1 = "cat perched on the sofa looking out of the window"
        text2 = "a bird fly over the sea"

        # -----------------------------Text1-------------------------------
        with StableDiffusionHooker(pipe) as hooker:
            set_seed(123456)
            out1 = pipe(prompt = tri + text1)
            image1 = out1.images[0]
        # 获取ovam优化后的trigger的embd的图
        embedding = ovam_evaluator.encode_text(
            text=tri,
            context_sentence=tri + text1,
            remove_special_tokens=False,
        )[:-1]  # This returns <Sot> + Cat + <Eot> tokens. Eot is removed.

        # Evaluate the attention map with the word cat and the optimized embedding
        with torch.no_grad():
            ovam_evaluator = hooker.get_ovam_callable(expand_size=(512, 512))
            optimized_map = ovam_evaluator(embedding).squeeze().cpu().numpy()[1] # (512, 512)
            non_optimized_map = ovam_evaluator("cat").squeeze().cpu().numpy()[1] # (512, 512)
        # binary_mask = densecrf(np.array(image1), optimized_map / optimized_map.max())
        binary_mask_1 = densecrf(np.array(image1), (optimized_map / optimized_map.max()) > 0.5)

        # -----------------------------Text2-------------------------------
        with StableDiffusionHooker(pipe) as hooker:
            set_seed(123456)
            out2 = pipe(prompt = tri + text1)
            image2 = out2.images[0]
        embedding = ovam_evaluator.encode_text(
            text=tri,
            context_sentence=text2,
            remove_special_tokens=False,
        )[:-1]  # This returns <Sot> + Cat + <Eot> tokens. Eot is removed.

        # Evaluate the attention map with the word cat and the optimized embedding
        with torch.no_grad():
            ovam_evaluator = hooker.get_ovam_callable(expand_size=(512, 512))
            optimized_map = ovam_evaluator(embedding).squeeze().cpu().numpy()[1] # (512, 512)
            non_optimized_map = ovam_evaluator("cat").squeeze().cpu().numpy()[1] # (512, 512)
        # binary_mask = densecrf(np.array(image1), optimized_map / optimized_map.max())
        binary_mask_2 = densecrf(np.array(image2), (optimized_map / optimized_map.max()) > 0.5)

        # -----------------------------Loss-------------------------------
        loss = loss_fn(binary_mask_1, binary_mask_2)
        loss.backward()
        optimizer.step()
        scheduler.step()

    return x.detach().cpu()

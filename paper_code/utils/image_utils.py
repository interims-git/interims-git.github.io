#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image or batch of images to YCbCr (BT.601).
    Input:  [3, H, W] or [N, 3, H, W], RGB in [0,1]
    Output: same shape, YCbCr in [0,1]
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)  # [1,3,H,W]
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]

    # BT.601 coefficients
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    return torch.cat((y, cb, cr), dim=1)  # [N,3,H,W]

def mse(img1, img2, per_image=False):
    assert img1.shape == img2.shape, "Inputs must have the same shape"
    diff = (img1 - img2) ** 2
    if per_image and img1.ndim >= 4:
        return diff.view(img1.shape[0], -1).mean(1, keepdim=True)
    return diff.mean()

@torch.no_grad()
def psnr_(img1, img2, mask=None):
    img1 = img1.flatten(1)
    img2 = img2.flatten(1)
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    if mask is not None:
        if torch.isinf(psnr).any():
            print('An inf ',mse.mean(),psnr.mean())
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
            psnr = psnr[~torch.isinf(psnr)]
        
    return psnr

@torch.no_grad()
def psnr(img1, img2, mask=None):

    if mask is not None:
        assert mask.shape == img1.shape[-2:], "Mask must match HxW of the image"
        mask = mask.expand_as(img1)
        diff = (img1 - img2) ** 2 * mask
        mse = diff.sum() / mask.sum()
    else:
        mse = ((img1 - img2) ** 2).mean()
    
    mse = torch.clamp(mse, min=1e-10)  # Prevent log(0)
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr_value
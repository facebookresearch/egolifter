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

def psnr(img1, img2, mask=None):
    if mask is None:
        mse = (((img1 - img2)) ** 2).mean()
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    else:
        mask = mask.to(img1).expand(img1.shape)
        img1 = img1 * mask
        img2 = img2 * mask
        mse = (((img1 - img2)) ** 2).sum() / mask.sum()
        return 20 * torch.log10(1.0 / torch.sqrt(mse))



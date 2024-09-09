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
from torch import nn
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from gaussian_grouping.utils.general_utils import PILtoTorch
from gaussian_grouping.utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, image_path, image_width, image_height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", objects=None, style_transfer=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = image_width
        self.image_height = image_height

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # self.image_width = image.shape[2]
        # self.image_height = image.shape[1]

        # if gt_alpha_mask is not None:
        #     self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if objects is not None:
            self.objects = objects.to(self.data_device)
        else:
            self.objects = None
        
        # if style_transfer:
        #     self.transfer_image = self.original_image.clone()
            
    def get_image(self, resolution=1, resolution_scale=1):
        image = Image.open(self.image_path)
        resolution = getResolution(resolution, image, resolution_scale)
        resized_image_rgb = PILtoTorch(image, resolution)

        # If the image has an alpha channel, extract it as the mask
        gt_image = resized_image_rgb[:3, ...]
        alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            alpha_mask = resized_image_rgb[3:4, ...]

        return gt_image


def getResolution(resolution, image: Image.Image | np.ndarray, resolution_scale: int) -> tuple[int, int]:
    if isinstance(image, np.ndarray):
        orig_w, orig_h = image.shape[1], image.shape[0]
    else:
        orig_w, orig_h = image.size

    if resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * resolution)), round(orig_h/(resolution_scale * resolution))
    else:  # should be a type that converts to float
        if resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    return resolution

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


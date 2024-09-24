# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import copy
import gzip
import pickle
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch import nn
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, focal2fov


class Camera(nn.Module):
    def __init__(
        self, colmap_id, uid, 
        R, T, FoVx, FoVy, 
        image_width, image_height,
        image_name, image_path, 
        scale = 1.0,
        camera_name = 'rgb', 
        fid = 0,
        scene_name = "",
        exposure: Optional[float] = 1.0, # the exposure duration (in second) of this image
        gain: Optional[float] = 1.0, # Gain of the image sensor
        seg_path: Optional[str] = None, # path to the segmentation mask
        valid_mask_subpath: Optional[str] = None, # path to the valid mask
        vignette_subpath: Optional[str] = None, # path to the vignette mask
        scene_folder: Optional[str] = None, # path to the scene folder
    ):
        super(Camera, self).__init__()

        self.uid = uid # This is mostly used as the index of the camera among all cameras. 
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height
        self.image_name = image_name
        self.image_path = image_path
        self.scene_name = scene_name
        self.fid = fid # The timestamp of this camera

        self.seg_path = seg_path
        self.valid_mask_subpath = valid_mask_subpath
        self.vignette_subpath = vignette_subpath
        
        self.exposure = exposure
        self.gain = gain
        self.scene_folder = scene_folder

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = np.array([0.0, 0.0, 0.0])
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_name = camera_name
        

    def set_image_size(self, new_width, new_height, keep_fovy=True):
        old_width = self.image_width
        old_height = self.image_height
        self.image_width = new_width
        self.image_height = new_height

        # Adjust the FoV according to the new image size
        if keep_fovy:
            ratio = new_width / new_height / (old_width / old_height)
            self.FoVx = np.arctan(np.tan(self.FoVx/2) * ratio) * 2
        else:
            ratio = new_height / new_width / (old_height / old_width)
            self.FoVy = np.arctan(np.tan(self.FoVy/2) * ratio) * 2

        self.set_fov(self.FoVx, self.FoVy)


    def set_fov(self, FoVx, FoVy):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, data_device='cuda'):
        # self.original_image = self.original_image.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        

    def copy(self):
        new_cam = copy.deepcopy(self)
        # new_cam.original_image = self.original_image
        return new_cam
        

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



def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.image_width,
        'height' : camera.image_height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FoVx, camera.image_height),
        'fx' : fov2focal(camera.FoVy, camera.image_width)
    }
    return camera_entry
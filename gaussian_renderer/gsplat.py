# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
from torch.nn import functional as F
from gsplat import rasterization
from scene.gaussian_model import GaussianModel

from utils.rigid_utils import from_homogenous, to_homogenous

def handle_deformation(
    pc: GaussianModel,
    d_xyz = None, d_rotation=None, d_scaling=None, is_6dof=False,
):
    # Handle the xyz deformation if given
    if d_xyz is None:
        means3D = pc.get_xyz
    else:
        if is_6dof:
            if torch.is_tensor(d_xyz) is False:
                means3D = pc.get_xyz
            else:
                means3D = from_homogenous(
                    torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
        else:
            means3D = pc.get_xyz + d_xyz

    rotations = pc.get_rotation if d_rotation is None else pc.get_rotation + d_rotation
    rotations = F.normalize(rotations, p=2, dim=1)
    
    ## scaling deformation leads to unstable training - disable it for now
    # print("before d_scaling:", pc.get_scaling.min(), pc.get_scaling.max())
    # scales = pc.get_scaling if d_scaling is None else pc.get_scaling + F.softplus(d_scaling)
    # print("after d_scaling:", scales.min(), scales.max())
    scales = pc.get_scaling

    opacity = pc.get_opacity

    return means3D, rotations, opacity, scales
    

def render(
        viewpoint_camera, 
        pc : GaussianModel, 
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        render_feature: bool = False,
        d_xyz = None, d_rotation=None, d_scaling=None, is_6dof=False,
    ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    means3D, rotations, opacity, scales = handle_deformation(pc, d_xyz, d_rotation, d_scaling, is_6dof)

    scales = scales * scaling_modifier
    
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    # First render the RGB color image
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=pc.get_features,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=pc.active_sh_degree,
    )

    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass


    # Then render the feature map if told so
    if render_feature:
        bg_feature = torch.zeros(pc.get_features_extra.shape[-1]).to(bg_color)
        render_features, _, info_feat = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=pc.get_features_extra,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_feature[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=None,
        )
        render_features = render_features[0].permute(2, 0, 1) # [D, H, W]
    else:
        render_features = None


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_features": render_features, # (D, H, W)
            "viewspace_points": info["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii,
            }
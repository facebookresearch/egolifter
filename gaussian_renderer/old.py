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

import math
from typing import Union
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.gaussian_union import GaussianUnion
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def render(
        viewpoint_camera, 
        pc : GaussianModel | GaussianUnion, 
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        render_feature: bool = False,
        d_xyz = None, d_rotation=None, d_scaling=None, is_6dof=False,
    ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        dim_features=pc.dim_extra if render_feature else 0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

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

    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # print("before d_scaling:", pc.get_scaling.min(), pc.get_scaling.max())
        # scales = pc.get_scaling if d_scaling is None else pc.get_scaling + F.softplus(d_scaling)
        # print("after d_scaling:", scales.min(), scales.max())
        scales = pc.get_scaling
        
        rotations = pc.get_rotation if d_rotation is None else pc.get_rotation + d_rotation
        rotations = F.normalize(rotations, p=2, dim=1)
        # rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if render_feature:
        features = pc.get_features_extra
    else:
        features = torch.zeros(
            (pc.get_features_extra.shape[0], 0), 
            dtype=pc.get_features.dtype, device=pc.get_features.device)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, render_features, radii, accum_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        features = features,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image, # (3, H, W)
            "render_features": render_features, # (D, H, W)
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "accum_alpha": accum_alpha,
            }

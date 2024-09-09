import os
import warnings
from typing import Any
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.utilities.types import STEP_OUTPUT

from scipy.spatial import KDTree

from utils.system_utils import get_last_ply_path
from utils.general_utils import freeze_module
from utils.graphics_utils import BasicPointCloud
from scene import GaussianModel, Scene, GaussianUnion
from gaussian_renderer.old import render

from .deform import DeformGaussian


class DeformStaticGaussian(DeformGaussian):
    '''
    Add a pretrained static Gaussian model to the deforming one. 
    '''
    def __init__(self, cfg, scene: Scene):
        super().__init__(cfg, scene)

        assert cfg.model.load_static_folder is not None, "The static model folder must be provided."
        assert os.path.exists(cfg.model.load_static_folder), f"The static model folder {cfg.model.load_static_folder} does not exist."

        # Load a pretrained Gaussian model as the static component of the scene
        static_ply_path = get_last_ply_path(cfg.model.load_static_folder)
        self.static_gaussians = GaussianModel(
            sh_degree=self.cfg.model.sh_degree,
            dim_extra=self.cfg.model.dim_extra,
        )
        self.static_gaussians.load_ply(static_ply_path)
        self.static_gaussians.eval() # This has no effect so far
        freeze_module(self.static_gaussians)
        
        print(f"Loaded the static Gaussian model of size {self.static_gaussians.get_xyz.shape[0]} from {static_ply_path}")
        
        
    
    def init_or_load_gaussians(
        self,
        init_point_cloud: BasicPointCloud,
        spatial_lr_scale: float,
        model_path: str, 
        load_iteration: int = None,
    ):
        # Filter out the points that are already explained by the static Gaussians
        static_xyz = self.static_gaussians.get_xyz.detach().cpu().numpy() # (N, 3)
        init_xyz = init_point_cloud.points # (M, 3)
        
        print(f"Before filtering, dynamic Gaussians have {init_xyz.shape[0]} points.")

        tree = KDTree(static_xyz)
        D, I = tree.query(init_xyz, k=1) # (M, ), (M, )
        remain_mask = D > self.cfg.model.filter_dist_thresh # (M, )
        
        init_point_cloud.points = init_point_cloud.points[remain_mask]
        init_point_cloud.colors = init_point_cloud.colors[remain_mask]
        init_point_cloud.normals = init_point_cloud.normals[remain_mask]
        
        print(f"After filtering, dynamic Gaussians have {init_point_cloud.points.shape[0]} points.")
        
        super().init_or_load_gaussians(
            init_point_cloud,
            spatial_lr_scale,
            model_path,
            load_iteration,
        )
        
        print(f"Initialized the dynamic Gaussian model of size {self.gaussians.get_xyz.shape[0]}")
        
    
    def forward(
        self, 
        viewpoint_cam, 
        render_feature=False, 
        fid = None,
        scaling_modifier=1.0, 
        no_static: bool = False, 
    ):
        if fid is None:
            fid = viewpoint_cam.fid
        
        # Compute the deformation for the dynamic part
        d_xyz, d_rot, d_scale, prob = self.forward_deform(self.gaussians, fid)
        
        if no_static:
            # Only visualize the deforming gaussians
            gaussians_union = self.gaussians
            d_xyz_union, d_rot_union, d_scale_union = d_xyz, d_rot, d_scale
        else:
            # Construct the union of both the static and dynamic Gaussians
            gaussians_union = GaussianUnion([self.static_gaussians, self.gaussians])
            
            n_static = self.static_gaussians.get_xyz.shape[0]
            d_xyz_union = None if d_xyz is None else torch.cat([torch.zeros(n_static, 3, device=self.device), d_xyz], dim=0)
            d_rot_union = None if d_rot is None else torch.cat([torch.zeros(n_static, 4, device=self.device), d_rot], dim=0)
            d_scale_union = None if d_scale is None else torch.cat([torch.ones(n_static, 3, device=self.device), d_scale], dim=0)
        
        render_pkg = render(
            viewpoint_cam, 
            gaussians_union, 
            self.cfg.pipe, 
            self.background, 
            scaling_modifier=scaling_modifier,
            render_feature=render_feature,
            # Deformation parameters
            d_xyz=d_xyz_union, 
            d_rotation=d_rot_union, 
            d_scaling=d_scale_union, 
            is_6dof=self.is_6dof,
        )
        
        render_pkg['d_xyz'] = d_xyz
        render_pkg['d_rot'] = d_rot
        render_pkg['d_scale'] = d_scale
        render_pkg['prob'] = prob
        render_pkg["no_static"] = no_static

        return render_pkg
    
    def density_control(self, outputs: STEP_OUTPUT) -> None:
        iteration = self.train_iter
        render_pkg = outputs["render_pkg"]
        render_pkg_feat = outputs["render_pkg_feat"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_point_tensor = render_pkg["viewspace_points"]

        if not render_pkg["no_static"]:
            # Adjust the tensors used for density control
            visibility_filter = visibility_filter[self.static_gaussians.get_xyz.shape[0]:]
            radii = radii[self.static_gaussians.get_xyz.shape[0]:]
        
        # Densification
        if iteration < self.cfg.opt.densify_until_iter:
            # # Keep track of max radii in image-space for pruning
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Use the gradient from both rendering images and rendering features
            grad = viewspace_point_tensor.grad
            if not render_pkg["no_static"]:
                grad = grad[self.static_gaussians.get_xyz.shape[0]:]

            vs_grad_norm = torch.norm(grad[visibility_filter, :2], dim=-1, keepdim=True)
            if render_pkg_feat is None:
                vs_grad_norm = vs_grad_norm
            else:
                grad = render_pkg_feat['viewspace_points'].grad
                if not render_pkg["no_static"]:
                    grad = grad[self.static_gaussians.get_xyz.shape[0]:]
                    
                vs_grad_norm_feat = torch.norm(grad[visibility_filter, :2], dim=-1, keepdim=True)
                vs_grad_norm = vs_grad_norm + self.cfg.opt.densify_grad_feat_scale * vs_grad_norm_feat
                
            self.gaussians.add_densification_stats(vs_grad_norm, visibility_filter)

            if iteration > self.cfg.opt.densify_from_iter and iteration % self.cfg.opt.densification_interval == 0:
                size_threshold = 20 if iteration > self.cfg.opt.opacity_reset_interval else None
                self.gaussians.densify_and_prune(self.cfg.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
            
            if iteration % self.cfg.opt.opacity_reset_interval == 0 or (self.cfg.model.white_background and iteration == self.cfg.opt.densify_from_iter):
                self.gaussians.reset_opacity()

        self.log_dict({
            'gaussians/total_points': float(self.gaussians.get_xyz.shape[0])
        }, on_step=True, on_epoch=False, logger=True)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco


import warnings
import torch
import numpy as np
from torch import nn
import os

import wandb
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData, PlyElement

from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH, SH2RGB
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.optimizer import (
    replace_tensor_to_optimizer,
    prune_optimizer_by_mask,
    cat_tensors_to_optimizer,
)


def knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm, actual_covariance
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, dim_extra: int = 0):
        super().__init__()
        # Constants config
        self.max_sh_degree = sh_degree  
        self.dim_extra = dim_extra
        self.percent_dense = 0

        # Optimizable Parameters
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._features_extra = None # Shape should be (num_points, dim_extra)
        self._scaling = None
        self._rotation = None
        self._opacity = None
        
        # Non-optimizable Buffers
        self.register_buffer("active_sh_degree", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("spatial_lr_scale", torch.tensor(0))

        self.register_buffer("max_radii2D", torch.zeros((0)))
        self.register_buffer("xyz_gradient_accum", torch.zeros((0, 1)))
        self.register_buffer("denom", torch.zeros((0, 1)))

        # Optimizer
        self.optimizer = None
        
        self.setup_functions()
        
    def check_nan(self):
        nan_xyz = torch.isnan(self._xyz).sum()
        nan_features_dc = torch.isnan(self._features_dc).sum()
        nan_features_rest = torch.isnan(self._features_rest).sum()
        nan_features_extra = torch.isnan(self._features_extra).sum()
        nan_scaling = torch.isnan(self._scaling).sum()
        nan_rotation = torch.isnan(self._rotation).sum()
        nan_opacity = torch.isnan(self._opacity).sum()
        
        flag = False
        if nan_xyz > 0:
            print("Nan in xyz")
            flag = True
        if nan_features_dc > 0:
            print("Nan in features_dc")
            flag = True
        if nan_features_rest > 0:
            print("Nan in features_rest")
            flag = True
        if nan_features_extra > 0:
            print("Nan in features_extra")
            flag = True
        if nan_scaling > 0:
            print("Nan in scaling")
            flag = True
        if nan_rotation > 0:
            print("Nan in rotation")
            flag = True
        if nan_opacity > 0:
            print("Nan in opacity")
            flag = True
            
        if flag:
            print("Nan in model")
            exit(0)


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc # (N, 1, 3)
    
    def get_rgb(self):
        return SH2RGB(self.get_features_dc)
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_features_extra(self):
        return self._features_extra
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)[0]
    
    def get_covariance_matrix(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)[1]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def create_from_size(self, n_points: int):
        '''
        Initialize the parameters and buffers from n_points. Needed before loading a state_dict
        '''
        self._xyz = nn.Parameter(torch.zeros((n_points, 3)).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.zeros((n_points, 1, 3)).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.zeros((n_points, (self.max_sh_degree + 1) ** 2 - 1, 3)).requires_grad_(True))
        self._features_extra = nn.Parameter(torch.zeros((n_points, self.dim_extra)).requires_grad_(True))
        self._scaling = nn.Parameter(torch.zeros((n_points, 3)).requires_grad_(True))
        self._rotation = nn.Parameter(torch.zeros((n_points, 4)).requires_grad_(True))
        self._opacity = nn.Parameter(torch.zeros((n_points, 1)).requires_grad_(True))
        
        self.max_radii2D = torch.zeros((n_points))
        self.xyz_gradient_accum = torch.zeros((n_points, 1))
        self.denom = torch.zeros((n_points, 1))

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = torch.tensor(spatial_lr_scale).to(self.spatial_lr_scale)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # Make features_extra initialized from a small normal distribution
        features_extra = 1e-3 * torch.normal(
            mean=0.0, std=1.0, 
            size=(fused_color.shape[0], self.dim_extra)
        ).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2_avg = (knn(torch.from_numpy(np.asarray(pcd.points)).float().cuda(), 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_extra = nn.Parameter(features_extra.contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.color_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.color_lr / 20.0, "name": "f_rest"},
            {'params': [self._features_extra], 'lr': training_args.feature_lr, "name": "f_extra"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, skip_extra=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        if not skip_extra:
            for i in range(self._features_extra.shape[1]):
                l.append('f_extra_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, skip_extra=False): # skip_extra to make it useable by the renderer
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_extra = self._features_extra.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(skip_extra=skip_extra)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if skip_extra:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, f_extra, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = replace_tensor_to_optimizer(self.optimizer, opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # Load SH coefficients
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        rest_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        rest_f_names = sorted(rest_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(rest_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_rest = np.zeros((xyz.shape[0], len(rest_f_names)))
        for idx, attr_name in enumerate(rest_f_names):
            features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_rest = features_rest.reshape((features_rest.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # Load extra features
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_extra_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        if self.dim_extra == 0:
            if len(extra_f_names) > 0:
                warnings.warn("Extra features found in the ply file, but the model was initialized without extra features")
            features_extra = np.zeros((xyz.shape[0], 0))
        else:
            assert len(extra_f_names)==self.dim_extra, "Expected {} extra features, found {}".format(self.dim_extra, len(extra_f_names))
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_extra = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Re-initialize the max radii
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = torch.tensor(self.max_sh_degree)

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        else:
            self._xyz.grad = None
            self._features_dc.grad = None
            self._features_rest.grad = None
            self._features_extra.grad = None
            self._opacity.grad = None
            self._scaling.grad = None
            self._rotation.grad = None


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer_by_mask(self.optimizer, valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_extra = optimizable_tensors["f_extra"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        

    def mask_points(self, valid_points_mask):
        assert self.optimizer is None, "Masking points is not supported during training"
        with torch.no_grad():
            self._xyz = nn.Parameter(self._xyz[valid_points_mask]).requires_grad_(True)
            self._features_dc = nn.Parameter(self._features_dc[valid_points_mask]).requires_grad_(True)
            self._features_rest = nn.Parameter(self._features_rest[valid_points_mask]).requires_grad_(True)
            self._features_extra = nn.Parameter(self._features_extra[valid_points_mask]).requires_grad_(True)
            self._opacity = nn.Parameter(self._opacity[valid_points_mask]).requires_grad_(True)
            self._scaling = nn.Parameter(self._scaling[valid_points_mask]).requires_grad_(True)
            self._rotation = nn.Parameter(self._rotation[valid_points_mask]).requires_grad_(True)
            self.max_radii2D = self.max_radii2D[valid_points_mask]


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_features_extra, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "f_extra": new_features_extra,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = cat_tensors_to_optimizer(self.optimizer, d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_extra = optimizable_tensors["f_extra"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        grad_mask = torch.where(padded_grad >= grad_threshold, True, False)
        scaling_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent

        selected_pts_mask = torch.logical_and(
            grad_mask,
            scaling_mask
        )
        wandb.log({"gaussians/densify_n_split": selected_pts_mask.sum().item()}, commit=False)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_features_extra = self._features_extra[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features_extra, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        wandb.log({"gaussians/densify_n_clone": selected_pts_mask.sum().item()}, commit=False)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_features_extra = self._features_extra[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features_extra, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        wandb.log({"gaussians/prune_n_opacity": prune_mask.sum().item()}, commit=False)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            wandb.log({"gaussians/prune_n_vs_radii": big_points_vs.sum().item()}, commit=False)
            wandb.log({"gaussians/prune_n_ws_radii": big_points_ws.sum().item()}, commit=False)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, vs_grad_norm, update_filter):
        self.xyz_gradient_accum[update_filter] += vs_grad_norm
        self.denom[update_filter] += 1
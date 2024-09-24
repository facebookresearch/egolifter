# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import wandb

# from .networks.deform import DeformNetwork
from scene import GaussianModel
from utils.general_utils import get_expon_lr_func, get_linear_noise_func
from gaussian_renderer.gsplat import render
from utils.image_utils import psnr

from .networks.deform2 import DeformNetwork
from .vanilla import VanillaGaussian


class DeformGaussian(VanillaGaussian):
    '''
    Learn a deformation field to warp the base 3DGS accoding to the time
    '''
    def __init__(self, cfg, scene):
        super().__init__(cfg, scene)
        
        self.is_blender = cfg.model.is_blender
        self.is_6dof = cfg.model.is_6dof
        self.noisy_fid_training = cfg.model.noisy_fid_training
        
        self.deform = DeformNetwork(
            D=cfg.model.net_depth, 
            W=cfg.model.net_width,
        )
        
        self.automatic_optimization = False

        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        
        
    def setup(self, stage: str):
        super().setup(stage)
        
        # Set up the optimizer and scheduler for the deform network here
        if stage == "fit":
            l = [
                {'params': list(self.deform.parameters()),
                'lr': self.cfg.opt.position_lr_init,
                "name": "deform"}
            ]
            self.optimizer_deform = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            
            self.scheduler_deform = get_expon_lr_func(
                lr_init=self.cfg.opt.position_lr_init,
                lr_final=self.cfg.opt.position_lr_final,
                lr_delay_mult=self.cfg.opt.position_lr_delay_mult,
                max_steps=self.cfg.opt.deform_lr_max_steps
            )
            
            
    def update_deform_lr(self, iteration):
        for param_group in self.optimizer_deform.param_groups:
            if param_group["name"] == "deform":
                lr = self.scheduler_deform(iteration)
                param_group['lr'] = lr
                return lr
        
        
    def forward_deform(
        self, 
        gaussians: GaussianModel, 
        fid: float | torch.Tensor
    ):
        if fid is None:
            d_xyz, d_rot, d_scale = None, None, None
        else:
            if fid < 0 or fid > 1:
                warnings.warn(f"fid should be in [0, 1], but got {fid}.")
            N = gaussians.get_xyz.shape[0]
            x_input = gaussians.get_xyz.detach()
            t_input = (torch.ones(N, 1, device=self.device) * fid).float()
            
            d_xyz, d_rot, d_scale, prob = self.deform(x_input, t_input)
            
            # Scale the deformation by the probability
            if self.cfg.model.use_prob:
                d_xyz = d_xyz * prob
                d_rot = d_rot * prob
                d_scale = d_scale * prob

            if not self.cfg.model.use_d_xyz:
                d_xyz = None
            if not self.cfg.model.use_d_rot:
                d_rot = None
            if not self.cfg.model.use_d_scale:
                d_scale = None
        
        return d_xyz, d_rot, d_scale, prob
        
            
    def forward(
        self, 
        viewpoint_cam, 
        render_feature=False, 
        fid = None,
        scaling_modifier=1.0
    ):
        '''
        Render the frame according to the Camera and time (fid)

        Args:
            - viewpoint_cam: Camera, the viewpoint of the camera
            - render_feature: bool, whether to render the feature
            - fid: float, the time used for deformation. If None, the deformation is not used
        '''
        if fid is None:
            fid = viewpoint_cam.fid
        
        d_xyz, d_rot, d_scale, prob = self.forward_deform(self.gaussians, fid)
        
        render_pkg = render(
            viewpoint_cam, 
            self.gaussians, 
            self.cfg.pipe, 
            self.background, 
            scaling_modifier=scaling_modifier,
            render_feature=render_feature,
            # Deformation parameters
            d_xyz=d_xyz, 
            d_rotation=d_rot, 
            d_scaling=d_scale, 
            is_6dof=self.is_6dof,
        )
        
        render_pkg['d_xyz'] = d_xyz
        render_pkg['d_rot'] = d_rot
        render_pkg['d_scale'] = d_scale
        render_pkg['prob'] = prob
        return render_pkg
    
    
    def forward_and_compute_loss(self, batch, render_func=None):
        gt_image = batch['image'].to("cuda")[0] # (3, H, W)
        subset = batch['subset'][0]
        viewpoint_cam = self.scene.get_camera(batch['idx'].item(), subset=subset)
        viewpoint_cam.image_width = gt_image.shape[2]
        viewpoint_cam.image_height = gt_image.shape[1]

        if render_func is None: render_func = self
        render_pkg = render_func(viewpoint_cam)
        image_rendered = render_pkg["render"]
        image_processed, gt_image_processed = self.scene.postProcess(image_rendered, gt_image, viewpoint_cam)

        losses = self.compute_recon_loss(image_processed, gt_image_processed)

        # Compute the PSNR with the valid pixel masks
        with torch.no_grad():
            losses['psnr'] = psnr(
                image_processed, gt_image_processed, batch["valid_mask"][0]
            ).mean().double()

        render_pkg_feat = None
        if self.use_contrast and viewpoint_cam.camera_name == "rgb":
            loss_contr, render_pkg_feat = self.forward_and_contr_loss(batch, render_func=render_func)
            losses['loss_mask_contr'] = loss_contr
            losses['loss_total'] = losses['loss_total'] + self.cfg.lift.lambda_contr * loss_contr

        # Regularization on the deformation offset
        if self.cfg.model.weight_l1_reg_prob > 0:
            losses['loss_l1_reg_prob'] = render_pkg["prob"].abs().mean()
            losses['loss_total'] = losses['loss_total'] + self.cfg.model.weight_l1_reg_prob * losses['loss_l1_reg_prob']
            
        if self.cfg.model.weight_l1_reg_xyz > 0:
            losses['loss_l1_reg_xyz'] = render_pkg["d_xyz"].abs().mean()
            losses['loss_total'] = losses['loss_total'] + self.cfg.model.weight_l1_reg_xyz * losses['loss_l1_reg_xyz']
            
        if self.cfg.model.weight_l1_reg_rot > 0:
            losses['loss_l1_reg_rot'] = render_pkg["d_rot"].abs().mean()
            losses['loss_total'] = losses['loss_total'] + self.cfg.model.weight_l1_reg_rot * losses['loss_l1_reg_rot']
            
        outputs = {
            "losses": losses,
            "image_processed": image_processed,
            "gt_image_processed": gt_image_processed,
            "render_pkg": render_pkg,
            "render_pkg_feat": render_pkg_feat,
        }
        
        return outputs
    

    def training_step(self, batch, batch_idx):
        # Handle fid, the timestamp for deformation
        fid = batch['fid'][0]
        if self.noisy_fid_training:
            time_interval = 1.0 / len(self.trainer.train_dataloader)
            fid = fid + torch.randn(1).to(fid) * time_interval * self.smooth_term(self.train_iter)
        
        render_func = partial(self, fid=fid)
        outputs = self.forward_and_compute_loss(batch, render_func=render_func)
        losses = outputs["losses"]
        render_pkg = outputs["render_pkg"]
        render_pkg_feat = outputs["render_pkg_feat"]

        logs = {f"train/{k}": v for k, v in losses.items()}

        # Also log the histogram of deform network output
        if self.train_iter % 200 == 0:
            extra_log = {}
            if render_pkg['prob'] is not None:
                hist = wandb.Histogram(render_pkg['prob'].detach().cpu().numpy(), num_bins=50)
                extra_log['deform/prob'] = hist
            if render_pkg['d_xyz'] is not None:
                log10_d_xyz = np.log10(render_pkg['d_xyz'].flatten().detach().abs().cpu().numpy() + 1e-6)
                hist = wandb.Histogram(log10_d_xyz, num_bins=100)
                extra_log['deform/log10_d_xyz'] = hist
            if render_pkg['d_rot'] is not None:
                log10_d_rot = np.log10(render_pkg['d_rot'].flatten().detach().abs().cpu().numpy() + 1e-6)
                hist = wandb.Histogram(log10_d_rot, num_bins=100)
                extra_log['deform/log10_d_rot'] = hist
            self.logger.experiment.log(extra_log)

        self.log_dict(logs, on_step=True, on_epoch=True, logger=True, batch_size=1)
        
        to_return = {
            "loss": losses["loss_total"],
            "render_pkg": render_pkg,
            "render_pkg_feat": render_pkg_feat,
        }

        # Manual optimization
        opt_gaussians, opt_deform = self.optimizers()
        opt_gaussians.zero_grad(set_to_none=True)
        opt_deform.zero_grad(set_to_none=True)

        self.manual_backward(losses["loss_total"])

        opt_gaussians.step()
        if self.train_iter >= self.cfg.model.opt_deform_start_iter:
            opt_deform.step()
    
        # This is needed to avoid prevent overlarge reserved memory and OOM.
        # torch.cuda.empty_cache()
        
        return to_return
    
        
    def configure_optimizers(self):
        optimizer_gaussian = super().configure_optimizers()
        optimizer_deform = self.optimizer_deform
        
        return [optimizer_gaussian, optimizer_deform]
        
        
        
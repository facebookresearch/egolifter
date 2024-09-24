# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import os
from argparse import Namespace
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import wandb

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from sklearn.decomposition import PCA

from scene.dataset_readers import BasicPointCloud, storePly
from scene import Scene

# Switch to gsplat implementation
from gaussian_renderer.gsplat import render
from scene.gsplat_model import GsplatModel as GaussianModel

from utils.loss_utils import l1_loss, l2_loss, ssim
from utils.image_utils import psnr
from utils.vis import concate_images_vis, feat_image_to_pca_vis
from utils.system_utils import searchForMaxIteration
from utils.general_utils import to_scalar


def compute_recon_metrics(image, gt_image, mask = None):
    L_l1 = l1_loss(image, gt_image, mask=mask).mean().double()
    L_l2 = l2_loss(image, gt_image, mask=mask).mean().double()
    metric_psnr = psnr(image, gt_image, mask=mask).mean().double()
    
    metrics = {
        "L_l1": L_l1,
        "L_l2": L_l2,
        "psnr": metric_psnr,
    }
    
    return metrics

class VanillaGaussian(L.LightningModule):
    def __init__(self, cfg: DictConfig, scene: Scene=None):
        super().__init__()
        self.cfg = cfg
            
        bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Init the Gaussians Model
        self.gaussians = GaussianModel(
            self.cfg.model.sh_degree, 
            self.cfg.model.dim_extra,
        )
        
        self.scene = scene
        
        self.contrast_manager = scene.contrast_manager if scene is not None else None
        self.use_contrast = self.contrast_manager is not None and self.contrast_manager.in_use
            
    def save_point_cloud(self, save_path: str):
        self.gaussians.save_ply(save_path)
        
    def init_gaussians_size_from_state_dict(self, state_dict: Dict[str, Any]):
        for k, v in state_dict.items():
            if k.endswith("._xyz"):
                module = self
                for name in k.split(".")[:-1]:
                    module = getattr(module, name)
                assert isinstance(module, GaussianModel)
                module.create_from_size(v.shape[0])

                # Additionally handle the buffer tensors, which could be of a different sizes
                module_name = ".".join(k.split(".")[:-1])
                for name in ["denom", "xyz_gradient_accum"]:
                    buffer_state_dict = state_dict[f"{module_name}.{name}"]
                    setattr(module, name, torch.zeros_like(buffer_state_dict))
                
    
    def init_or_load_gaussians(
        self,
        init_point_cloud: BasicPointCloud,
        spatial_lr_scale: float,
        model_path: str, 
        load_iteration: int = None,
    ):
        loaded_iter = None
        if load_iteration:
            if load_iteration == -1:
                loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
            else:
                loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(loaded_iter))

        if loaded_iter:
            self.gaussians.load_ply(os.path.join(model_path,
                                            "point_cloud",
                                            "iteration_" + str(loaded_iter),
                                            "point_cloud.ply"),
                                og_number_points=len(init_point_cloud.points))
        else:
            self.gaussians.create_from_pcd(init_point_cloud, spatial_lr_scale)
            
        if not loaded_iter:
            storePly(
                os.path.join(model_path, "input.ply"), 
                init_point_cloud.points,
                init_point_cloud.colors * 255,
            )
            
            
    def setup(self, stage: str):
        '''
        Set up the trainable parameters and optimizer for the GaussianModel
        setup() ensures this is called before configure_optimizers()
        '''
        if stage == "fit":
            # Create and set up the output folder
            print("Output folder: {}".format(self.cfg.scene.model_path))
            os.makedirs(self.cfg.scene.model_path, exist_ok = True)
            
            # Save configs. TODO: Change this to a yaml file
            cfg_args_save_path = os.path.join(self.cfg.scene.model_path, "cfg_args")
            if not os.path.exists(cfg_args_save_path):
                with open(cfg_args_save_path, 'w') as cfg_log_f:
                    cfg_log_f.write(str(Namespace(**vars(self.cfg))))
            
            print("Setting up for training")
            self.gaussians.training_setup(self.cfg.opt)
            
            
    def on_train_start(self) -> None:
        '''
        Log scene information, the initial point cloud and camera positions
        '''
        if self.scene is not None:
            if self.cfg.log_cam_stats:
                # Log the statistics of the scene
                logs = self.scene.get_scene_cam_time()
                for log in logs: wandb.log(log, commit=True)
            
            # Log the point cloud with the training camera positions
            point_cloud = self.scene.scene_info.point_cloud.points # (M, 3)
            point_cloud = np.concatenate([point_cloud, np.ones_like(point_cloud) * 128], axis=1) # (M, 6), color with gray
            
            camera_poses = self.scene.get_camera_poses() # (N, 4, 4)
            camera_points = camera_poses[:, :3, 3] # (N, 3)
            camera_points = np.concatenate([camera_points, np.ones_like(camera_points) * np.array([[255,0,0]])], axis=1) # (N, 6), color with red
            point_cloud = np.concatenate([point_cloud, camera_points], axis=0) # (M+N, 6)
        else:
            # Log the gaussian point cloud as is
            point_cloud = self.gaussians.get_xyz.detach().cpu().numpy()  # (N, 3)
            rgb = (self.gaussians.get_rgb().detach().cpu().squeeze().numpy() * 255).round()  # (N, 3)
            point_cloud = np.concatenate([point_cloud, rgb], axis=1) # (N, 6)
        
        # The Chirality of wandb visualizer is different from open3d, so we need to flip one of the axis
        point_cloud[:, 1] *= -1

        point_scene = wandb.Object3D({
            "type": "lidar/beta",
            "points": point_cloud
        })
        wandb.log({"gaussian_vis/init_point_cloud": point_scene}, commit=True)
        
        self.train_iter = 0
        
        
    def on_train_end(self) -> None:
        '''
        Log the final point cloud
        '''
        final_point_cloud = self.gaussians.get_xyz.detach().cpu().numpy()  # (N, 3)
        final_rgb = (self.gaussians.get_rgb().detach().cpu().squeeze().numpy() * 255).round()  # (N, 3)
        final_point_cloud = np.concatenate([final_point_cloud, final_rgb], axis=1) # (N, 6)
        final_point_cloud[:, 1] *= -1
        point_scene = wandb.Object3D({
            "type": "lidar/beta",
            "points": final_point_cloud
        })
        wandb.log({"gaussian_vis/final_point_cloud": point_scene}, commit=True)


    def forward(
        self, 
        viewpoint_cam, 
        render_feature=False, 
        fid=None,
        scaling_modifier=1.0
    ):
        '''
        Render the scene from the viewpoint_cam
        '''
        if fid is not None:
            warnings.warn("fid is passed in, but it will be ignored in VanillaGaussian.forward()!")
            
        render_pkg = render(
            viewpoint_cam, 
            self.gaussians, 
            self.cfg.pipe, 
            self.background, 
            scaling_modifier=scaling_modifier,
            render_feature=render_feature
        )
        return render_pkg
    
    
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        '''
        Update Gaussians learning rate and SH degree
        '''
        self.train_iter += 1
        self.log("train_iter", float(self.train_iter), on_step=True, on_epoch=True, logger=True, batch_size=1)
        
        self.gaussians.update_learning_rate(self.train_iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.train_iter % 1000 == 0:
            self.gaussians.oneupSHdegree()
            
    
    def get_active_sh_degree(self):
        return self.gaussians.active_sh_degree
    
    
    def set_active_sh_degree(self, degree):
        assert degree < self.gaussians.max_sh_degree
        self.gaussians.active_sh_degree = degree
        
    
    def get_max_sh_degree(self):
        return self.gaussians.max_sh_degree
            
        
    def compute_recon_loss(self, image: torch.Tensor, gt_image: torch.Tensor) -> dict:
        L_l1 = l1_loss(image, gt_image).mean().double()
        L_ssim = 1.0 - ssim(image, gt_image)
        loss = (1.0 - self.cfg.opt.lambda_dssim) * L_l1 + self.cfg.opt.lambda_dssim * L_ssim

        # with torch.no_grad():
        #     psnr_batch = psnr(image, gt_image).mean()
            
        return {
            "loss_total": loss,
            "loss_l1": L_l1,
            "loss_ssim": L_ssim,
            # "psnr": psnr_batch,
        }
        
    def forward_and_contr_loss(self, batch, weight_image=None, render_func=None):
        mask_image = batch['mask_image'][0]
        subset = batch['subset'][0]
        viewpoint_cam = self.scene.get_camera(batch['idx'].item(), subset=subset)
        viewpoint_cam_mask = self.contrast_manager.get_feat_cam(viewpoint_cam)

        if render_func is None: render_func = self
        render_pkg_feat = render_func(viewpoint_cam_mask, render_feature=True)
        render_features = render_pkg_feat['render_features'] # (D, H, W)

        # Extract the last a few channels as the instance feature field
        render_features = render_features[-self.cfg.lift.contr_dim:, :, :] # (D, H, W)
        render_features = render_features.permute(1, 2, 0).contiguous() # (H, W, D)
        
        if weight_image is not None:
            # Input weight image should be (1, H, W)
            assert weight_image.shape[0] == 1 and weight_image.dim() == 3, f"weight_image.shape: {weight_image.shape}"
            # Resize the weight image to the same size as the render features
            weight_image = F.interpolate(
                weight_image.unsqueeze(0), 
                size=render_features.shape[:2],
                mode='bilinear', 
                align_corners=False
            ).squeeze(0) # (1, H, W)

        loss_contr = self.contrast_manager.compute_loss(
            mask_image, 
            render_features, 
            temperature=self.cfg.lift.temperature,
            weight_image=weight_image,
        )

        return loss_contr, render_pkg_feat
    
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
            
        outputs = {
            "losses": losses,
            "image_processed": image_processed,
            "gt_image_processed": gt_image_processed,
            "render_pkg": render_pkg,
            "render_pkg_feat": render_pkg_feat,
        }
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward_and_compute_loss(batch)
        losses = outputs["losses"]
        render_pkg = outputs["render_pkg"]
        render_pkg_feat = outputs["render_pkg_feat"]

        logs = {f"train/{k}": v for k, v in losses.items()}
        self.log_dict(logs, on_step=True, on_epoch=True, logger=True, batch_size=1)

        to_return = {
            "loss": losses["loss_total"],
            "render_pkg": render_pkg,
            "render_pkg_feat": render_pkg_feat,
        }
        
        return to_return
    
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.gaussians.check_nan()
        self.density_control(outputs)


    def density_control(self, outputs: STEP_OUTPUT) -> None:
        iteration = self.train_iter
        render_pkg = outputs["render_pkg"]
        render_pkg_feat = outputs["render_pkg_feat"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        
        # Densification
        if iteration < self.cfg.opt.densify_until_iter:
            # # Keep track of max radii in image-space for pruning
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            image = render_pkg["render"]
            self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

            # # Use the gradient from both rendering images and rendering features
            # vs_grad_norm = torch.norm(viewspace_point_tensor.grad[visibility_filter, :2], dim=-1, keepdim=True)

            # # if render_pkg_feat is None or render_pkg_feat['viewspace_points'].grad is None:
            # #     pass
            # # else:
            # #     vs_grad_norm_feat = torch.norm(render_pkg_feat['viewspace_points'].grad[visibility_filter, :2], dim=-1, keepdim=True)
            # #     vs_grad_norm = vs_grad_norm + self.cfg.opt.densify_grad_feat_scale * vs_grad_norm_feat

            # self.gaussians.add_densification_stats(vs_grad_norm, visibility_filter)

            if iteration > self.cfg.opt.densify_from_iter and iteration % self.cfg.opt.densification_interval == 0:
                size_threshold = 20 if iteration > self.cfg.opt.opacity_reset_interval else None
                self.gaussians.densify_and_prune(self.cfg.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
            
            if iteration % self.cfg.opt.opacity_reset_interval == 0 or (self.cfg.model.white_background and iteration == self.cfg.opt.densify_from_iter):
                self.gaussians.reset_opacity()

        self.log_dict({
            'gaussians/total_points': float(self.gaussians.get_xyz.shape[0])
        }, on_step=True, on_epoch=False, logger=True)
        

    def configure_optimizers(self):
        return self.gaussians.optimizer
    
    
    def optimizers(self, use_pl_optimizer: bool = True):
        optimizers = super().optimizers(use_pl_optimizer=use_pl_optimizer)

        if isinstance(optimizers, list) is False:
            return [optimizers]

        """
        IMPORTANCE: the global_step will be increased on every step() call of all the optimizers,
        issue https://github.com/Lightning-AI/lightning/issues/17958,
        here change _on_before_step and _on_after_step to override this behavior.
        """
        for idx, optimizer in enumerate(optimizers):
            if idx == 0:
                continue
            optimizer._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
            optimizer._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")

        return optimizers
    
    
    def log_image(
            self, 
            batch, 
            image_rendered, 
            image_processed, 
            gt_image, 
            feat_image=None
        ):
        subset = batch['subset'][0]
        
        viewpoint_cam = self.scene.get_camera(batch['idx'].item(), subset=subset)
        log_name = subset + "_view/{}".format(viewpoint_cam.image_name)
            
        render_wandb = (image_rendered.permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8)
        image_wandb = (image_processed.permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8)
        gt_image_wandb = (gt_image.permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8)

        # Stack images horizontally
        images_wandb = [gt_image_wandb, image_wandb, render_wandb]

        if feat_image is not None:
            feat_image_wandb = (feat_image * 255).astype(np.uint8)
            images_wandb.append(feat_image_wandb)
            
        image_wandb = concate_images_vis(images_wandb)
        # image_wandb = np.concatenate(images_wandb, axis=1)
        image_wandb = wandb.Image(image_wandb, caption="GT : Render (training) : Render (direct)")
        wandb.log({log_name: image_wandb}, commit=True)
        
    
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        with torch.no_grad():
            outputs = self.forward_and_compute_loss(batch)
            
        losses = outputs["losses"]
        image_rendered = outputs["render_pkg"]["render"].clamp(0.0, 1.0)
        image_processed = outputs["image_processed"].clamp(0.0, 1.0)
        gt_image_processed = outputs["gt_image_processed"].clamp(0.0, 1.0)
        render_pkg_feat = outputs["render_pkg_feat"]

        dynamic_mask = None
        if "dynamic_mask" in batch:
            dynamic_mask = batch["dynamic_mask"][0]

        num_val_batches = self.trainer.num_val_batches[dataloader_idx] if isinstance(self.trainer.num_val_batches, list) else self.trainer.num_val_batches
        log_image_indices = np.linspace(0, num_val_batches - 1, 20, dtype=int)
        if batch_idx in log_image_indices:
            # Get a PCA visualization of the features
            feat_image = None
            if render_pkg_feat is not None:
                feat_image = render_pkg_feat['render_features'].detach() # (D, H, W)
                feat_image = feat_image_to_pca_vis(feat_image, channel_first=True)
                
            self.log_image(
                batch, 
                image_rendered, 
                image_processed, 
                gt_image_processed, 
                feat_image=feat_image
            )
            
        logs = {f"valid_loader_{dataloader_idx}/{k}": v for k, v in losses.items()}

        if dynamic_mask is not None:
            dynamic_mask = batch["dynamic_mask"][0]
            if dynamic_mask.sum() > 0:
                dyna_metrics = compute_recon_metrics(image_processed, gt_image_processed, dynamic_mask)
                logs.update({f"valid_loader_{dataloader_idx}_metrics/dyna_{k}": v for k, v in dyna_metrics.items()})

            static_mask = batch["static_mask"][0]
            if static_mask.sum() > 0:
                static_metrics = compute_recon_metrics(image_processed, gt_image_processed, static_mask)
                logs.update({f"valid_loader_{dataloader_idx}_metrics/static_{k}": v for k, v in static_metrics.items()})
            
        self.log_dict(logs, on_step=True, on_epoch=True, logger=True, batch_size=1, add_dataloader_idx=False)
        
        
    def on_test_epoch_start(self) -> None:
        self.test_logs = []
    
    def test_step(self, batch, batch_idx):
        '''
        The test step will collect more information during the evaluation
        '''
        with torch.no_grad():
            outputs = self.forward_and_compute_loss(batch)
            
        losses = outputs["losses"]
        subset = batch['subset'][0]
        
        logs = {f"eval_on_{subset}/{k}": v for k, v in losses.items()}

        # Log the information about the image
        test_log = {
            "exp_name": self.cfg.exp_name,
            "scene_name": batch['scene_name'][0],
            "image_name": batch['image_name'][0],
            "image_id": batch['image_id'][0].item(),
            "subset": subset,
        }
        
        if "dynamic_mask" in batch:
            image_processed = outputs["image_processed"].clamp(0.0, 1.0)
            gt_image_processed = outputs["gt_image_processed"].clamp(0.0, 1.0)
            dynamic_mask = batch["dynamic_mask"][0]
            static_mask = batch["static_mask"][0]

            if dynamic_mask.sum() > 0:
                dyna_metrics = compute_recon_metrics(image_processed, gt_image_processed, dynamic_mask)
                logs.update({f"eval_on_{subset}/dyna_{k}": v for k, v in dyna_metrics.items()})
                test_log.update({f"dyna_{k}": v for k, v in dyna_metrics.items()})

            if static_mask.sum() > 0:
                static_metrics = compute_recon_metrics(image_processed, gt_image_processed, static_mask)
                logs.update({f"eval_on_{subset}/static_{k}": v for k, v in static_metrics.items()})
                test_log.update({f"static_{k}": v for k, v in static_metrics.items()})
        
        # Log to the logger
        self.log_dict(logs, on_step=True, on_epoch=True, logger=True, batch_size=1)
        
        # Log to the csv file
        for k, v in losses.items():
            test_log[k] = to_scalar(v)
            
        for k, v in test_log.items():
            if torch.is_tensor(v):
                test_log[k] = to_scalar(v)
                
        self.test_logs.append(test_log)
    
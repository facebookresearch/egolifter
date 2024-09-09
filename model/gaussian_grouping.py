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

from scene import GaussianModel, Scene

from .vanilla import compute_recon_metrics

from utils.loss_utils import l1_loss, l2_loss, ssim
from utils.gaussian_grouping import parse_namespace
from utils.image_utils import psnr
from utils.general_utils import to_scalar

from gaussian_grouping.scene import Scene as gg_Scene
from gaussian_grouping.gaussian_renderer import GaussianModel as gg_GaussianModel
from gaussian_grouping.gaussian_renderer import render as gg_render


class GaussianGrouping(L.LightningModule):
    def __init__(self, cfg: DictConfig, scene: Scene=None):
        super().__init__()
        self.cfg = cfg
        self.scene = scene
        
        assert cfg.model.gg_ckpt_folder is not None, "gg_ckpt_folder is not set"

        ckpt_args_path = os.path.join(cfg.model.gg_ckpt_folder, "cfg_args")
        # Load the dataset arguments and overwrite the model path and source path
        dataset_args = Namespace(**parse_namespace(ckpt_args_path))
        dataset_args.model_path = cfg.model.gg_ckpt_folder
        dataset_args.source_path = cfg.scene.source_path
        
        # Construct the pipeline arguments
        pipeline_args = Namespace(
            convert_SHs_python = False,
            compute_cov3D_python = False,
            debug = False,
        )
        gaussians = gg_GaussianModel(dataset_args.sh_degree)
        gg_scene = gg_Scene(dataset_args, gaussians, load_iteration=-1, shuffle=False)
        
        num_classes = dataset_args.num_classes
        
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(
            cfg.model.gg_ckpt_folder, "point_cloud","iteration_"+str(gg_scene.loaded_iter),"classifier.pth"))
        )
        bg_color = [1,1,1] if dataset_args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        self.pipeline_args = pipeline_args
        self.gaussians = gaussians
        self.gg_scene = gg_scene
        self.num_classes = num_classes
        self.classifier = classifier
        # self.networks = {}
        # self.networks['classifier'] = classifier
        self.background = background
        
    def forward(
        self, 
        viewpoint_cam, 
        render_feature=True, 
        fid=None,
        scaling_modifier=1.0
    ):
        '''
        Render the scene from the viewpoint_cam
        '''
        render_pkg = gg_render(viewpoint_cam, self.gaussians, self.pipeline_args, self.background)
        
        # For compatibility with the vanilla model
        render_pkg["render_features"] = render_pkg["render_object"]
        
        return render_pkg
    
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
        
    def forward_and_compute_loss(self, batch):
        gt_image = batch['image'].to("cuda")[0] # (3, H, W)
        subset = batch['subset'][0]
        viewpoint_cam = self.scene.get_camera(batch['idx'].item(), subset=subset)
        viewpoint_cam.image_width = gt_image.shape[2]
        viewpoint_cam.image_height = gt_image.shape[1]

        # if render_func is None: render_func = self
        # render_pkg = render_func(viewpoint_cam)
        # image_rendered = render_pkg["render"]
        render_pkg = gg_render(viewpoint_cam, self.gaussians, self.pipeline_args, self.background)
        image_rendered = render_pkg["render"]
        
        image_processed, gt_image_processed = self.scene.postProcess(image_rendered, gt_image, viewpoint_cam)

        losses = self.compute_recon_loss(image_processed, gt_image_processed)
        
        # Compute the PSNR with the valid pixel masks
        with torch.no_grad():
            losses['psnr'] = psnr(
                image_processed, gt_image_processed, batch["valid_mask"][0]
            ).mean().double()

        render_pkg_feat = None
        
        # if self.use_contrast and viewpoint_cam.camera_name == "rgb":
        #     loss_contr, render_pkg_feat = self.forward_and_contr_loss(batch, render_func=render_func)
        #     losses['loss_mask_contr'] = loss_contr
        #     losses['loss_total'] = losses['loss_total'] + self.cfg.lift.lambda_contr * loss_contr

        
        
        outputs = {
            "losses": losses,
            "image_processed": image_processed,
            "gt_image_processed": gt_image_processed,
            "render_pkg": render_pkg,
            "render_pkg_feat": render_pkg_feat,
        }
        
        return outputs
    
    
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
        
        
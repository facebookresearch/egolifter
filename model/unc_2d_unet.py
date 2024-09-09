import math
from typing import Any
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from backbones_unet.model.unet import Unet

import wandb

from .vanilla import VanillaGaussian, compute_recon_metrics

from utils.vis import concate_images_vis, feat_image_to_pca_vis
from utils.image_utils import psnr

class Unc2DUnet(VanillaGaussian):
    '''
    Learning an image based uncertainty map, which is used to guide 3D reconstruction. 
    The map should have high values for transient regions and low values for static regions.
    '''
    def __init__(self, cfg, scene):
        super().__init__(cfg, scene)

        print("Initializing U-Net...")
        self.net_conf = Unet(
            backbone=cfg.model.unet_backbone, # backbone network name
            in_channels=3,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
            num_classes=1,            # output channels (number of classes in your dataset)
            pretrained=cfg.model.unet_pretrained,          # use ImageNet pretrained weights
            preprocessing=True,       # Apply the training mean and std normalization
            activation='identity',    # TODO: figure out a proper activation function
        )  # (1, 3, H, W) -> (1, 1, H, W)
        
        # Manual optimizeration
        self.automatic_optimization = False
        
        assert self.cfg.model.unet_acti in ['softplus', 'sigmoid', 'baseline'], f"Unknown activation function {self.cfg.model.unet_acti}"
        assert self.cfg.model.recon_loss not in ['3dgs'], "3dgs loss does not work and is deprecated. "
        assert self.cfg.model.contr_weight_mode in ['thresh', "on_sim", "on_prob"], f"Unknown contrastive weight mode {self.cfg.model.contr_weight_mode}"

        
    def compute_recon_loss(self, image: torch.Tensor, gt_image: torch.Tensor, mask: torch.Tensor) -> dict:
        # The loss used in NeRF-W
        if self.cfg.model.unet_acti == "softplus":
            loss_l2_unc = ( (image - gt_image).pow(2) / (2.0 * mask.pow(2)) ).mean()
            loss_unc_reg = ( torch.log(mask.pow(2)) / 2.0 ).mean()
        elif self.cfg.model.unet_acti == "sigmoid":
            # mask is treated as the probability of being transient
            loss_l2_unc = ( (image - gt_image).pow(2) * (1.0 - mask) ).mean()
            if self.cfg.model.unc_reg_loss == "l1":
                loss_unc_reg = mask.abs().mean()
            elif self.cfg.model.unc_reg_loss == "l2":
                loss_unc_reg = F.mse_loss(mask, torch.zeros_like(mask))
            else:
                raise ValueError(f"Unknown regularization loss {self.cfg.model.unc_reg_loss}")
        elif self.cfg.model.unet_acti == "baseline":
            loss_l2_unc = (image - gt_image).pow(2).mean()
            loss_unc_reg = 0.0
            
        loss = loss_l2_unc + self.cfg.model.weight_loss_reg * loss_unc_reg
        
        losses = {
            "loss_total": loss,
            "loss_l2_unc": loss_l2_unc,
            "loss_unc_reg": loss_unc_reg,
        }

        # if self.cfg.model.recon_loss == "l2":
        #     loss_l2_unc = F.mse_loss(image, gt_image)
        #     loss = loss_l2_unc + self.cfg.model.weight_loss_reg * loss_unc_reg

        #     losses["loss_l2_unc"] = loss_l2_unc
        #     losses["loss_total"] = loss
        # elif self.cfg.model.recon_loss == "3dgs":
        #     losses_3dgs = super().compute_recon_loss(image, gt_image)
        #     loss_total = losses_3dgs["loss_total"] + self.cfg.model.weight_loss_reg * loss_unc_reg
        #     losses.update(losses_3dgs)
        #     losses["loss_total"] = loss_total # Overwrite the loss_total with the new value with regularization

        # for k, v in losses.items():
        #     if torch.isnan(v):
        #         import pdb; pdb.set_trace()
        
        return losses
        
    def get_unc_mask(self, batch) -> torch.Tensor:
        if self.cfg.model.unet_acti == "baseline":
            return None
        
        gt_image = batch['image'].to("cuda")[0].clamp(0.0, 1.0) # (3, H, W)
        
        H, W = gt_image.shape[-2:]
        # Get the uncertainty map by feeding GT image to the U-Net
        unet_input = gt_image.unsqueeze(0) # (1, 3, H, W)
        unet_input = F.interpolate(
            unet_input, 
            size=(self.cfg.model.unet_input_size, self.cfg.model.unet_input_size), 
            mode='bilinear', 
            align_corners=False
        ) # (1, 3, H', W')

        unet_input = torch.rot90(unet_input, -1, [2, 3]) # (1, 3, H', W'), rot 90 deg clockwise
        unet_output = self.net_conf(unet_input) # (1, 1, H', W')
        unet_output = torch.rot90(unet_output, 1, [2, 3]) # (1, 1, H', W'), rot 90 deg counter-clockwise
        
        mask = F.interpolate(unet_output, size=(H, W), mode='bilinear', align_corners=False) # (1, 1, H, W)
        mask = mask.squeeze(0) # (1, H, W)

        if self.cfg.model.unet_acti == "sigmoid":
            mask = torch.sigmoid(mask) # (1, H, W)
        elif self.cfg.model.unet_acti == "softplus":
            mask = F.softplus(mask) + self.cfg.model.unc_min # (1, H, W)
            
        return mask
    
    def forward_and_compute_loss(self, batch):
        subset = batch['subset'][0]
        gt_image = batch['image'].to("cuda")[0].clamp(0.0, 1.0) # (3, H, W)
        viewpoint_cam = self.scene.get_camera(batch['idx'].item(), subset=subset)
        viewpoint_cam.image_width = gt_image.shape[2]
        viewpoint_cam.image_height = gt_image.shape[1]
        
        render_pkg = self(viewpoint_cam)
        image_rendered = render_pkg["render"]
        
        unc_mask = self.get_unc_mask(batch)

        image_processed, gt_image_processed = self.scene.postProcess(image_rendered, gt_image, viewpoint_cam)
        
        # Compute the loss and log
        losses = self.compute_recon_loss(image_processed, gt_image_processed, unc_mask)
        
        # Compute the PSNR with the valid pixel masks
        with torch.no_grad():
            losses['psnr'] = psnr(
                image_processed, gt_image_processed, batch["valid_mask"][0]
            ).mean().double()

        render_pkg_feat = None
        if self.use_contrast and viewpoint_cam.camera_name == "rgb":
            static_mask = None if unc_mask is None else 1.0 - unc_mask 
            loss_contr, render_pkg_feat = self.forward_and_contr_loss(batch, weight_image=static_mask)
            losses['loss_mask_contr'] = loss_contr
            losses['loss_total'] = losses['loss_total'] + self.cfg.lift.lambda_contr * loss_contr

        outputs = {
            "losses": losses,
            "unc_mask": unc_mask,
            "image_processed": image_processed,
            "gt_image_processed": gt_image_processed,
            "render_pkg": render_pkg,
            "render_pkg_feat": render_pkg_feat
        }
        
        return outputs
        
    def training_step(self, batch, batch_idx):
        outputs = self.forward_and_compute_loss(batch)
        losses = outputs["losses"]
        render_pkg = outputs["render_pkg"]
        render_pkg_feat = outputs["render_pkg_feat"]
        
        # if self.train_iter % 1000 == 0:
        #     self.log_image(batch, image_rendered.detach(), image_processed.detach(), gt_image, unc_mask.detach(), image_name="render")
        
        logs = {f"train/{k}": v for k, v in losses.items()}
        self.log_dict(logs, on_step=True, on_epoch=True, logger=True, batch_size=1)

        to_return = {
            "loss": losses["loss_total"],
            "render_pkg": render_pkg,
            "render_pkg_feat": render_pkg_feat
        }
        
        # Manual optimization
        for optimizer in self.optimizers():
            optimizer.zero_grad(set_to_none=True)

        self.manual_backward(losses["loss_total"])

        for optimizer in self.optimizers():
            optimizer.step()
        
        return to_return
        
    
    def configure_optimizers(self):
        optimizer_gaussian = super().configure_optimizers()
        optimizer_unet = torch.optim.Adam(self.net_conf.parameters(), lr=self.cfg.model.unet_lr)
        
        return [optimizer_gaussian, optimizer_unet]
    
    def log_image(
            self, 
            batch, 
            image_rendered, 
            image_processed, 
            gt_image, 
            unc_mask = None, 
            image_name = None,
            gt_mask = None,
            feat_image = None,
        ):
        subset = batch['subset'][0]
        
        if image_name is None:
            viewpoint_cam = self.scene.get_camera(batch['idx'].item(), subset=subset)
            image_name = viewpoint_cam.image_name
        log_name = subset + "_view/{}".format(image_name)

        render_wandb = (image_rendered.clamp(0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8)
        image_wandb = (image_processed.clamp(0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8)
        gt_image_wandb = (gt_image.clamp(0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8)

        # unc_vis = (unc_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8) # (H, W)
        # unc_vis = cv2.applyColorMap(unc_vis, cv2.COLORMAP_TURBO)[:, :, ::-1] # (H, W, 3) in RGB
        # unc_vis = cv2.addWeighted(gt_image_wandb, 0.5, unc_vis, 0.5, 0) # (H, W, 3)
        
        if unc_mask is not None:
            unc_vis = unc_mask.squeeze(0).cpu().numpy() # (H, W)
            fig = plt.figure(dpi=200, figsize=(unc_vis.shape[1] / 200, unc_vis.shape[0] / 200))
            plt.imshow(gt_image_wandb)
            plt.imshow(unc_vis, cmap='turbo', alpha=0.5)
            plt.colorbar()
            plt.axis('off')
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            unc_vis = np.array(fig.canvas.renderer._renderer)[:, :, :3]
            fig.clf()
            plt.close(fig)
        else:
            unc_vis = np.zeros_like(gt_image_wandb)

        # Stack images horizontally
        images_wandb = [gt_image_wandb, image_wandb, render_wandb, unc_vis]
        caption = "GT : Render (training) : Render (direct) : uncertainty map"
        
        if feat_image is not None:
            feat_image_wandb = (feat_image * 255).astype(np.uint8)
            images_wandb.append(feat_image_wandb)

        if gt_mask is not None:
            gt_mask_wandb = (gt_mask.permute(1, 2, 0).repeat(1, 1, 3).contiguous().cpu().numpy() * 255).astype(np.uint8)
            images_wandb.append(gt_mask_wandb)
            caption = caption + " : GT dynamic mask"
            
        image_wandb = concate_images_vis(images_wandb)
        image_wandb = wandb.Image(image_wandb, caption = caption)
        wandb.log({log_name: image_wandb}, commit=True)
        
    
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        with torch.no_grad():
            outputs = self.forward_and_compute_loss(batch)
            
        losses = outputs["losses"]
        image_rendered = outputs['render_pkg']["render"].clamp(0.0, 1.0)
        image_processed = outputs["image_processed"].clamp(0.0, 1.0)
        gt_image_processed = outputs["gt_image_processed"].clamp(0.0, 1.0)
        unc_mask = outputs["unc_mask"]
        render_pkg_feat = outputs["render_pkg_feat"]

        dynamic_mask = None
        if "dynamic_mask" in batch:
            dynamic_mask = batch["dynamic_mask"][0]

        num_val_batches = self.trainer.num_val_batches[dataloader_idx] if isinstance(self.trainer.num_val_batches, list) else self.trainer.num_val_batches
        log_image_indices = np.linspace(0, num_val_batches - 1, 20, dtype=int)
        if batch_idx in log_image_indices:
            feat_image = None
            if render_pkg_feat is not None:
                feat_image = render_pkg_feat['render_features'].detach() # (D, H, W)
                feat_image = feat_image_to_pca_vis(feat_image, channel_first=True)
            
            self.log_image(
                batch, 
                image_rendered, 
                image_processed, 
                gt_image_processed, 
                unc_mask = unc_mask, 
                gt_mask = dynamic_mask,
                feat_image = feat_image,
            )

        logs = {f"valid_loader_{dataloader_idx}/{k}": v for k, v in losses.items()}
        
        if dynamic_mask is not None:
            if dynamic_mask.sum() > 0:
                dyna_metrics = compute_recon_metrics(image_processed, gt_image_processed, dynamic_mask)
                logs.update({f"valid_loader_{dataloader_idx}_metrics/dyna_{k}": v for k, v in dyna_metrics.items()})

            static_mask = batch["static_mask"][0]
            if static_mask.sum() > 0:
                static_metrics = compute_recon_metrics(image_processed, gt_image_processed, static_mask)
                logs.update({f"valid_loader_{dataloader_idx}_metrics/static_{k}": v for k, v in static_metrics.items()})
        
        self.log_dict(logs, on_step=True, on_epoch=True, logger=True, batch_size=1, add_dataloader_idx=False)

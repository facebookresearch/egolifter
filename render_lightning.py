import os
from glob import glob
from pathlib import Path
import time
from typing import Any
import cv2
import imageio

from natsort import natsorted

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import torch
import torchvision
import lightning as L
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip, clips_array

from scene.dataset_readers import SceneType
from model.unc_2d_unet import Unc2DUnet
from utils.routines import load_from_model_path
from utils.pca import compute_feat_pca_from_renders
from utils.vis import resize_image


@hydra.main(version_base=None, config_path="conf", config_name="render")
def main(cfg_render : DictConfig) -> None:
    model_path = cfg_render.model_path
    model, scene, cfg = load_from_model_path(
        model_path, source_path=cfg_render.source_path, simple_scene=True)
    
    save_root = model_path
    if cfg_render.eval_on_gg:
        from model.gaussian_grouping import GaussianGrouping
        from gaussian_grouping.render import visualize_obj
        assert cfg_render.gg_ckpt_folder is not None, "gg_ckpt_folder must be specified"
        print(f"Loading GaussainGrouping model from {cfg_render.gg_ckpt_folder}")
        cfg.model.name = "gaussian_grouping"
        cfg.model.gg_ckpt_folder = cfg_render.gg_ckpt_folder
        model = GaussianGrouping(cfg, scene)

        save_root = Path(cfg_render.gg_ckpt_folder)
    
    if cfg_render.no_render_feat:
        do_pca_render = False
    else:    
        do_pca_render = cfg.model.dim_extra > 0

    L.seed_everything(cfg.seed)
    
    # Downsample a subset of the data and compute a PCA in the space of rendered features
    if do_pca_render:
        pca = compute_feat_pca_from_renders(scene, cfg_render.render_subset, [model])
        pca = pca[0]
  
    loader = scene.get_data_loader(cfg_render.render_subset, shuffle=False)

    save_folder = os.path.join(save_root, f"render_{cfg_render.render_subset}")
    
    gt_folder = os.path.join(save_folder, "gt")
    os.makedirs(gt_folder, exist_ok=True)
    render_folder = os.path.join(save_folder, "render")
    os.makedirs(render_folder, exist_ok=True)
    if do_pca_render:
        feat_folder = os.path.join(save_folder, "feature")
        os.makedirs(feat_folder, exist_ok=True)
        
        if cfg_render.eval_on_gg:
            pred_obj_folder = os.path.join(save_folder, "pred_obj")
            os.makedirs(pred_obj_folder, exist_ok=True)
            
    if isinstance(model, Unc2DUnet):
        trans_mask_folder = os.path.join(save_folder, "unc_mask")
        os.makedirs(trans_mask_folder, exist_ok=True)

    print(f"Rendering to {save_folder}")
    render_times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for batch_idx, batch in tqdm(enumerate(loader), total = len(loader)):
        gt_image = batch['image'].to("cuda")[0] # (3, H, W)
        subset = batch['subset'][0]
        viewpoint_cam = scene.get_camera(batch['idx'].item(), subset=subset)
        
        with torch.no_grad():
            start.record()
            render_pkg = model(viewpoint_cam, render_feature = do_pca_render)
            end.record()
            torch.cuda.synchronize()
            render_times.append(start.elapsed_time(end))
            
            render, gt_image_processed = scene.postProcess(render_pkg["render"], gt_image, viewpoint_cam)
            render = render.clamp(0.0, 1.0) # (3, H, W)
            
            # print(render.shape)
            if scene.scene_type == SceneType.ARIA:
                render_vis = np.rot90(render.permute(1, 2, 0).cpu().numpy(), k=-1) # (H, W, 3)
            else:
                render_vis = render.permute(1, 2, 0).cpu().numpy()
            # print(render_vis.shape)
            render_vis = resize_image(render_vis, cfg_render.image_size)
            # print(render_vis.shape)
            render_vis = (render_vis * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(render_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"), render_vis)
            
            if do_pca_render:
                render_feat = render_pkg["render_features"].permute(1, 2, 0).cpu().numpy() # (H, W, D)
                feat_shape = render_feat.shape[:2]
                render_feat = render_feat.reshape(-1, render_feat.shape[-1]) # (N, D)
                render_pca = pca.transform(render_feat) # (N, 3)
                render_pca = render_pca.reshape(feat_shape[0], feat_shape[1], 3) # (H, W, 3)
                
                if scene.scene_type == SceneType.ARIA:
                    render_pca_vis = np.rot90(render_pca, k=-1) # (W, H, 3)
                else:
                    render_pca_vis = render_pca
                render_pca_vis = resize_image(render_pca_vis, cfg_render.image_size) 
                render_pca_vis = (render_pca_vis * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(feat_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"), render_pca_vis)

                if cfg_render.eval_on_gg:
                    logits = model.classifier(render_pkg["render_features"])
                    pred_obj = torch.argmax(logits,dim=0)
                    pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
                    if scene.scene_type == SceneType.ARIA:
                        pred_obj_mask = np.rot90(pred_obj_mask, k=-1) # (W, H, 3)
                    else:
                        pred_obj_mask = pred_obj_mask
                    pred_obj_mask = resize_image(pred_obj_mask, cfg_render.image_size)
                    imageio.imwrite(os.path.join(pred_obj_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"), pred_obj_mask)

        # Convert tensor to numpy and rotate
        gt_image_vis = gt_image_processed.permute(1, 2, 0).cpu().numpy()
        if scene.scene_type == SceneType.ARIA:
            gt_image_vis = np.rot90(gt_image_vis, k=-1)
        else:
            gt_image_vis = gt_image_vis
        gt_image_vis = resize_image(gt_image_vis, cfg_render.image_size)

        if isinstance(model, Unc2DUnet):
            with torch.no_grad():
                trans_mask = model.get_unc_mask(batch)
                if scene.scene_type == SceneType.ARIA:
                    trans_mask_vis = np.rot90(trans_mask.squeeze(0).cpu().numpy(), k=-1)
                else:
                    trans_mask_vis = trans_mask.squeeze(0).cpu().numpy()
            trans_mask_vis = resize_image(trans_mask_vis, cfg_render.image_size)
            trans_mask_vis = cv2.applyColorMap(255 - (trans_mask_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            trans_mask_vis = trans_mask_vis.astype(gt_image_vis.dtype) / 255.0
            trans_mask_vis = cv2.addWeighted(gt_image_vis, 0.5, trans_mask_vis, 0.5, 0)
            trans_mask_vis = (trans_mask_vis * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(trans_mask_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"), trans_mask_vis)

        gt_image_vis = (gt_image_vis * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(gt_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"), gt_image_vis)
        
        # # Instead of duplicate the images, create a soft link
        # src = viewpoint_cam.image_path
        # gt_ext = src.split(".")[-1]
        # dst = os.path.join(gt_folder, '{0:05d}'.format(batch_idx) + f".{gt_ext}")
        # if not os.path.exists(dst):
        #     os.symlink(src, dst)

        # torchvision.utils.save_image(render_rgb, os.path.join(render_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"))
        
    print(f"Average render time: {np.mean(render_times):.2f} ms")

    # Concatenate the saved images into a video
    print(f"Concating to video")
    gt_paths = natsorted(glob(os.path.join(gt_folder, f"*.{cfg_render.save_ext}")))
    clip_gt = ImageSequenceClip(gt_paths, fps=cfg_render.fps)
    render_paths = natsorted(glob(os.path.join(render_folder, f"*.{cfg_render.save_ext}")))
    clip_render = ImageSequenceClip(render_paths, fps=cfg_render.fps)
    clips = [[clip_gt, clip_render]]

    if do_pca_render:
        feat_paths = natsorted(glob(os.path.join(feat_folder, f"*.{cfg_render.save_ext}")))
        clip_feat = ImageSequenceClip(feat_paths, fps=cfg_render.fps)
        clips[0].append(clip_feat)
        
        if cfg_render.eval_on_gg:
            pred_obj_paths = natsorted(glob(os.path.join(pred_obj_folder, f"*.{cfg_render.save_ext}")))
            clip_pred_obj = ImageSequenceClip(pred_obj_paths, fps=cfg_render.fps)
            clips[0].append(clip_pred_obj)
        
    clip_combined = clips_array(clips)
    clip_combined.write_videofile(os.path.join(save_folder, f"combined.mp4"))
    


if __name__ == "__main__":
    main()
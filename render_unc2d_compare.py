# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from glob import glob
from typing import Any
import PIL
import cv2
from matplotlib import pyplot as plt

from natsort import natsorted

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import moviepy
from moviepy.editor import ImageSequenceClip
from sklearn.decomposition import PCA

import torch
import lightning as L
from tqdm import tqdm

from scene import Scene
from model.vanilla import VanillaGaussian

from utils.pca import FeatPCA, compute_feat_pca_from_renders
from utils.vis import resize_image
from utils.routines import load_from_model_path


def enhance_image(image: np.ndarray, factor: float) -> np.ndarray:
    img = PIL.Image.fromarray((image * 255).astype(np.uint8))
    converter = PIL.ImageEnhance.Color(img)
    img2 = converter.enhance(factor)
    return np.array(img2) / 255.0


@hydra.main(version_base=None, config_path="conf", config_name="render_unc2d_compare")
def main(cfg_render : DictConfig) -> None:
    
    model, scene, cfg = load_from_model_path(cfg_render.model_path)
    model_baseline, _, cfg = load_from_model_path(cfg_render.model_baseline_path)
    
    do_pca_render = cfg.model.dim_extra > 0

    L.seed_everything(cfg.seed)
    
    # Downsample a subset of the data and compute a PCA in the space of rendered features
    if do_pca_render:
        [pca, pca_baseline] = compute_feat_pca_from_renders(scene, cfg_render.render_subset, [model, model_baseline])
    
    # Load the full dataset and render the video
    loader = scene.get_data_loader(cfg_render.render_subset, shuffle=False)

    save_folder = os.path.join(cfg_render.model_path, f"render_{cfg_render.render_subset}")
    image_folder = os.path.join(save_folder, "compare")

    os.makedirs(image_folder, exist_ok=True)

    print(f"Rendering to {save_folder}")
    for batch_idx, batch in tqdm(enumerate(loader), total = len(loader)):
        gt_image = batch['image'].to("cuda")[0] # (3, H, W)
        
        subset = batch['subset'][0]
        viewpoint_cam = scene.get_camera(batch['idx'].item(), subset=subset)
        
        with torch.no_grad():
            render_pkg = model(viewpoint_cam, render_feature = do_pca_render)
            render = render_pkg["render"].clamp(0.0, 1.0)
            render, gt_image = scene.postProcess(render, gt_image, viewpoint_cam)
            
            render_pkg_baseline = model_baseline(viewpoint_cam, render_feature = do_pca_render)
            render_baseline = render_pkg_baseline["render"].clamp(0.0, 1.0)
            render_baseline, _ = scene.postProcess(render_baseline, gt_image, viewpoint_cam)
            
            mask = model.get_unc_mask(batch)
            
            if do_pca_render:
                # Transform the rendered features to PCA colors
                render_feat = render_pkg["render_features"].permute(1, 2, 0).cpu().numpy() # (H, W, D)
                feat_shape = render_feat.shape[:2]
                render_feat = render_feat.reshape(-1, render_feat.shape[-1]) # (N, D)
                render_pca = pca.transform(render_feat) # (N, 3)
                render_pca = render_pca.reshape(feat_shape[0], feat_shape[1], 3) # (H, W, 3)
                
                render_feat_baseline = render_pkg_baseline["render_features"].permute(1, 2, 0).cpu().numpy() # (H, W, D)
                feat_shape = render_feat_baseline.shape[:2]
                render_feat_baseline = render_feat_baseline.reshape(-1, render_feat_baseline.shape[-1]) # (N, D)
                render_baseline_pca = pca_baseline.transform(render_feat_baseline) # (N, 3)
                render_baseline_pca = render_baseline_pca.reshape(feat_shape[0], feat_shape[1], 3) # (H, W, 3)
        
        image_size = 768
        
        # Convert tensor to numpy and rotate
        gt_image_vis = gt_image.permute(1, 2, 0).cpu().numpy()
        gt_image_vis = np.rot90(gt_image_vis, k=-1)
        render_baseline_vis = np.rot90(render_baseline.permute(1, 2, 0).cpu().numpy(), k=-1)
        mask_vis = np.rot90(mask.squeeze(0).cpu().numpy(), k=-1)
        render_vis = np.rot90(render.permute(1, 2, 0).cpu().numpy(), k=-1)

        # Resize images
        gt_image_vis = resize_image(gt_image_vis, image_size)
        render_baseline_vis = resize_image(render_baseline_vis, image_size)
        mask_vis = resize_image(mask_vis, image_size)
        render_vis = resize_image(render_vis, image_size)
        
        # Convert mask to turbo colormap and overlay it on gt_image_vis
        mask_vis = cv2.applyColorMap(255 - (mask_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        mask_vis = mask_vis.astype(gt_image_vis.dtype) / 255.0
        gt_image_overlay_vis = cv2.addWeighted(gt_image_vis, 0.5, mask_vis, 0.5, 0)
        
        # Combine images into one
        combined_image = (
            np.hstack((gt_image_vis, render_baseline_vis)), 
            np.hstack((gt_image_overlay_vis, render_vis))
        )
        
        if do_pca_render:
            render_pca_vis = np.rot90(render_pca, k=-1)
            render_baseline_pca_vis = np.rot90(render_baseline_pca, k=-1)
            render_pca_vis = resize_image(render_pca_vis, image_size)
            render_baseline_pca_vis = resize_image(render_baseline_pca_vis, image_size)
            
            combined_image = (
                np.hstack((gt_image_vis, render_baseline_vis, render_baseline_pca_vis)), 
                np.hstack((gt_image_overlay_vis, render_vis, render_pca_vis))
            )
            
        combined_image = np.vstack(combined_image)

        combined_image = (combined_image * 255.0).astype(np.uint8)
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

        # Add titles to the images
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_image, 'GT', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_image, 'Render from Baseline', (image_size + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_image, 'Uncertainty mask from U-Net', (10, image_size + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_image, 'Render with uncertainty mask', (image_size + 10, image_size + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if do_pca_render:
            cv2.putText(combined_image, 'Feature PCA from Baseline', (2 * image_size + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_image, 'Feature PCA with uncertainty mask', (2 * image_size + 10, image_size + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Save the image
        cv2.imwrite(os.path.join(image_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"), combined_image)

        # if batch_idx > 100:
        #     break
        
    image_paths = natsorted(glob(os.path.join(image_folder, f"*.{cfg_render.save_ext}")))
    output_video_path = os.path.join(save_folder, f"compare.mp4")
    clip = ImageSequenceClip(image_paths, fps=20)
    clip.write_videofile(output_video_path, fps=20)
    


if __name__ == "__main__":
    main()
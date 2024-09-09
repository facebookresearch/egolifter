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

from model.unc_2d_unet import Unc2DUnet

from utils.pca import FeatPCA, compute_feat_pca_from_renders
from utils.vis import resize_image
from utils.routines import load_from_model_path


@hydra.main(version_base=None, config_path="conf", config_name="render_compare")
def main(cfg_render : DictConfig) -> None:
    models = list()
    model_1, scene, cfg = load_from_model_path(cfg_render.model_1_path, source_path=cfg_render.source_path)
    models.append(model_1)
    if cfg_render.model_2_path is not None:
        model_2, _, _ = load_from_model_path(cfg_render.model_2_path, source_path=cfg_render.source_path)
        models.append(model_2)
    if cfg_render.model_3_path is not None:
        model_3, _, _ = load_from_model_path(cfg_render.model_3_path, source_path=cfg_render.source_path)
        models.append(model_3)
    
    do_pca_render = cfg.model.dim_extra > 0

    L.seed_everything(cfg.seed)
    
    # Downsample a subset of the data and compute a PCA in the space of rendered features
    if do_pca_render:
        pcas = compute_feat_pca_from_renders(scene, cfg_render.render_subset, models)
    
    # Load the full dataset and render the video
    loader = scene.get_data_loader(cfg_render.render_subset, shuffle=False)

    save_folder = os.path.join(cfg_render.model_1_path, f"render_{cfg_render.render_subset}")
    image_folder = os.path.join(save_folder, "compare_3")

    os.makedirs(image_folder, exist_ok=True)

    print(f"Rendering to {save_folder}")
    for batch_idx, batch in tqdm(enumerate(loader), total = len(loader)):
        gt_image = batch['image'].to("cuda")[0] # (3, H, W)
        subset = batch['subset'][0]
        viewpoint_cam = scene.get_camera(batch['idx'].item(), subset=subset)
        
        image_grid = [[]]
        if do_pca_render:
            image_grid = [[], []]
        
        with torch.no_grad():
            for i in range(len(models)):
                model = models[i]
                
                render_pkg = model(viewpoint_cam, render_feature = do_pca_render)
                render, gt_image_processed = scene.postProcess(render_pkg["render"], gt_image, viewpoint_cam)
                render = render.clamp(0.0, 1.0)

                render_vis = np.rot90(render.permute(1, 2, 0).cpu().numpy(), k=-1)
                render_vis = resize_image(render_vis, cfg_render.image_size)
                image_grid[0].append(render_vis)
                
                if do_pca_render:
                    pca = pcas[i]
                    render_feat = render_pkg["render_features"].permute(1, 2, 0).cpu().numpy() # (H, W, D)
                    feat_shape = render_feat.shape[:2]
                    render_feat = render_feat.reshape(-1, render_feat.shape[-1]) # (N, D)
                    render_pca = pca.transform(render_feat) # (N, 3)
                    render_pca = render_pca.reshape(feat_shape[0], feat_shape[1], 3) # (H, W, 3)
                    
                    render_pca_vis = np.rot90(render_pca, k=-1)
                    render_pca_vis = resize_image(render_pca_vis, cfg_render.image_size)
                    image_grid[1].append(render_pca_vis)
                    
        # Convert tensor to numpy and rotate
        gt_image_vis = gt_image_processed.permute(1, 2, 0).cpu().numpy()
        gt_image_vis = np.rot90(gt_image_vis, k=-1)
        gt_image_vis = resize_image(gt_image_vis, cfg_render.image_size)

        image_grid[0].insert(0, gt_image_vis)
        
        if do_pca_render:
            # If the first model is full model, also visualize the uncertainty mask
            bottom_left = np.zeros_like(gt_image_vis)
            if isinstance(models[0], Unc2DUnet):
                with torch.no_grad():
                    mask = models[0].get_unc_mask(batch)
                    mask_vis = np.rot90(mask.squeeze(0).cpu().numpy(), k=-1)
                mask_vis = resize_image(mask_vis, cfg_render.image_size)
                mask_vis = cv2.applyColorMap(255 - (mask_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                mask_vis = mask_vis.astype(gt_image_vis.dtype) / 255.0
                bottom_left = cv2.addWeighted(gt_image_vis, 0.5, mask_vis, 0.5, 0)
        
            image_grid[1].insert(0, bottom_left)
            
        # combined_image = np.vstack(combined_image)
        combined_image = np.vstack([np.hstack(row) for row in image_grid])

        combined_image = (combined_image * 255.0).astype(np.uint8)
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

        # Add titles to the images
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_image, 'GT', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(combined_image, 'Uncertainty mask from U-Net', (10, cfg_render.image_size + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_image, f'Render by {cfg_render.model_1_name}', (cfg_render.image_size + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_image, f'Render by {cfg_render.model_2_name}', (2 * cfg_render.image_size + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_image, f'Render by {cfg_render.model_3_name}', (3 * cfg_render.image_size + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if do_pca_render:
            if bottom_left.any():
                cv2.putText(combined_image, f'Transient map from {cfg_render.model_1_name}', (10, cfg_render.image_size + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_image, f'Feature PCA by {cfg_render.model_1_name}', (cfg_render.image_size + 10, cfg_render.image_size + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_image, f'Feature PCA by {cfg_render.model_2_name}', (2 * cfg_render.image_size + 10, cfg_render.image_size + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_image, f'Feature PCA by {cfg_render.model_3_name}', (3 * cfg_render.image_size + 10, cfg_render.image_size + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Save the image
        cv2.imwrite(os.path.join(image_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"), combined_image)

        # if batch_idx > 100:
        #     break
        
    image_paths = natsorted(glob(os.path.join(image_folder, f"*.{cfg_render.save_ext}")))
    output_video_path = os.path.join(save_folder, f"compare_3.mp4")
    clip = ImageSequenceClip(image_paths, fps=20)
    clip.write_videofile(output_video_path, fps=20)
    


if __name__ == "__main__":
    main()
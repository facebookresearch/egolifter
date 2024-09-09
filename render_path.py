import copy
import json
import os
from glob import glob
from pathlib import Path
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

from model.unc_2d_unet import Unc2DUnet
from utils.routines import load_from_model_path
from utils.pca import compute_feat_pca_from_renders
from utils.vis import resize_image


@hydra.main(version_base=None, config_path="conf", config_name="render_path")
def main(cfg_render : DictConfig) -> None:
    # Load the custom camera trajectories from the json file
    assert cfg_render.camera_path is not None
    cameras_meta = json.load(open(cfg_render.camera_path))
    orientation_transform = torch.tensor(cameras_meta["orientation_transform"], dtype=torch.float)
    cameras = cameras_meta["camera_path"]
    cameras = [np.asarray(c["camera_to_world"]).reshape((4,4)) for c in cameras]
    cameras = np.stack(cameras, axis=0)
    
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
    
    do_pca_render = cfg.model.dim_extra > 0

    L.seed_everything(cfg.seed)
    
    # Downsample a subset of the data and compute a PCA in the space of rendered features
    if do_pca_render:
        pca = compute_feat_pca_from_renders(scene, "train", [model])
        pca = pca[0]
  
    save_folder = os.path.join(save_root, f"render_path")
    
    render_folder = os.path.join(save_folder, "render")
    os.makedirs(render_folder, exist_ok=True)
    if do_pca_render:
        feat_folder = os.path.join(save_folder, "feature")
        os.makedirs(feat_folder, exist_ok=True)
        
        if cfg_render.eval_on_gg:
            pred_obj_folder = os.path.join(save_folder, "pred_obj")
            os.makedirs(pred_obj_folder, exist_ok=True)
            
    # Get an example camera for the scene
    viewpoint_cam_template = scene.get_camera(0, subset="train")
            
    print(f"Rendering to {save_folder}")
    
    for batch_idx, cam_matrix in tqdm(enumerate(cameras), total = len(cameras)):
        viewpoint_cam = copy.deepcopy(viewpoint_cam_template)

        # Transform the camera matrix from viser saved format to that in the dataset
        c2w = torch.tensor(cam_matrix, dtype=torch.float).unsqueeze(0)
        c2w = torch.matmul(orientation_transform, c2w)
        c2w[..., :3, 1:3] *= -1
        w2c = torch.linalg.inv(c2w)
        w2c = w2c.numpy()
        R = w2c[0, :3, :3]
        t = w2c[0, :3, 3]
        
        # Transform to the format used by the renderer
        R = R.T # row major to column major
        
        viewpoint_cam.reset_extrinsic(R, t)
        
        with torch.no_grad():
            render_pkg = model(viewpoint_cam, render_feature = do_pca_render)
            gt_image = torch.zeros_like(render_pkg["render"])
            render, gt_image_processed = scene.postProcess(render_pkg["render"], gt_image, viewpoint_cam)
            render = render.clamp(0.0, 1.0) # (3, H, W)
            
            # print(render.shape)
            render_vis = np.rot90(render.permute(1, 2, 0).cpu().numpy(), k=-1) # (H, W, 3)
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
                
                render_pca_vis = np.rot90(render_pca, k=-1) # (W, H, 3)
                render_pca_vis = resize_image(render_pca_vis, cfg_render.image_size) 
                render_pca_vis = (render_pca_vis * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(feat_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"), render_pca_vis)

                if cfg_render.eval_on_gg:
                    logits = model.classifier(render_pkg["render_features"])
                    pred_obj = torch.argmax(logits,dim=0)
                    pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
                    pred_obj_mask = np.rot90(pred_obj_mask, k=-1) # (W, H, 3)
                    pred_obj_mask = resize_image(pred_obj_mask, cfg_render.image_size)
                    imageio.imwrite(os.path.join(pred_obj_folder, '{0:05d}'.format(batch_idx) + f".{cfg_render.save_ext}"), pred_obj_mask)


    # Concatenate the saved images into a video
    print(f"Concating to video")
    render_paths = natsorted(glob(os.path.join(render_folder, f"*.{cfg_render.save_ext}")))
    clip_render = ImageSequenceClip(render_paths, fps=cfg_render.fps)
    clips = [[clip_render]]

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
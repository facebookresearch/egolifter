from dataclasses import dataclass
import os
from glob import glob
from pathlib import Path
import time
from typing import Any, Optional
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
import tyro
import hdbscan
from sklearn.cluster import KMeans

from utils.routines import load_from_model_path


@dataclass
class Main:
    model_path: str
    source_path: Optional[str] = None
    
    feat_subset: str = "all" # The subset to obtain sampled features
    render_subset: str = "valid"
    
    n_sample_views: int = 100
    n_sample_pixels: int = int(1e5)

    n_clusters: int = 20 # Only used for KMeans

    
    def main(self) -> None:
        model, scene, cfg = load_from_model_path(
            self.model_path, source_path=self.source_path, simple_scene=True)
        
        L.seed_everything(cfg.seed)
        
        save_root = self.model_path
        loader_feat = scene.get_data_loader(self.feat_subset, shuffle=True)
        loader = scene.get_data_loader(self.render_subset, shuffle=False)
        
        print(f"loader_feat: {len(loader_feat)}")
        print(f"loader: {len(loader)}")
        
        # Get sampled features
        sample_feat_all = []
        n_sample_per_img = self.n_sample_pixels // self.n_sample_views
        for batch_idx, batch in tqdm(enumerate(loader_feat), total = len(loader)):
            # At most sample n_sample_views images
            if batch_idx >= self.n_sample_views:
                break
            
            subset = batch['subset'][0]
            viewpoint_cam = scene.get_camera(batch['idx'].item(), subset=subset)
        
            with torch.no_grad():
                render_pkg = model(viewpoint_cam, render_feature = True)
                render_feat = render_pkg["render_features"].permute(1, 2, 0).cpu().numpy() # (H, W, D)
                render_feat = render_feat.reshape(-1, render_feat.shape[-1]) # (H*W, D)
                sample_feat = render_feat[np.random.choice(render_feat.shape[0], n_sample_per_img, replace=False)]
                sample_feat_all.append(render_feat)
        
        sample_feat_all = np.concatenate(sample_feat_all, axis=0)
        print(f"sample_feat_all: {sample_feat_all.shape}")
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(sample_feat_all)
        cluster_centers = kmeans.cluster_centers_
        
        # Iterate over the dataset and render all images
        for batch_idx, batch in tqdm(enumerate(loader), total = len(loader)):
            subset = batch['subset'][0]
            viewpoint_cam = scene.get_camera(batch['idx'].item(), subset=subset)
            
            with torch.no_grad():
                render_pkg = model(viewpoint_cam, render_feature = True)
                render_feat = render_pkg["render_features"].permute(1, 2, 0).cpu().numpy() # (H, W, D)

            # Assign render_feat to clusters for each pixel
            h, w = render_feat.shape[:2]
            render_feat = render_feat.reshape(-1, render_feat.shape[-1])
            render_class = kmeans.predict(render_feat)
            render_class = render_class.reshape(h, w)
            
            print(render_class.shape)
            print(render_class.dtype)
            print(render_class.min(), render_class.max())

            
            break
            
        
    
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Main).main()
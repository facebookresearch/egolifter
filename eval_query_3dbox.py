# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import json

from pathlib import Path
from glob import glob
from typing import Optional
from natsort import natsorted

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scipy.spatial import KDTree

import torch
import tyro

from model import get_model
from scene import Scene
from utils.eval_2dseg import compute_sampled_features
from utils.routines import load_from_model_path


import wandb

def bbox_iou_3d(aabb1: np.ndarray, aabb2: np.ndarray):
    '''
    Compute the IoU of two 3D bounding boxes.
    
    Args:
        aabb1: (6, ) array, [x1, y1, z1, x2, y2, z2] of the first bounding box
        aabb2: (6, ) array, [x1, y1, z1, x2, y2, z2] of the second bounding box
    
    Returns:
        iou: float
    '''
    # Compute the coordinates of the intersection of two bounding boxes
    x1 = max(aabb1[0], aabb2[0])
    y1 = max(aabb1[1], aabb2[1])
    z1 = max(aabb1[2], aabb2[2])
    x2 = min(aabb1[3], aabb2[3])
    y2 = min(aabb1[4], aabb2[4])
    z2 = min(aabb1[5], aabb2[5])
    # Compute the dimensions of the intersection
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    d = max(0, z2 - z1)
    # Compute the volume of the intersection
    inter_vol = w * h * d
    # Compute the volume of each bounding box
    vol_aabb1 = (aabb1[3] - aabb1[0]) * (aabb1[4] - aabb1[1]) * (aabb1[5] - aabb1[2])
    vol_aabb2 = (aabb2[3] - aabb2[0]) * (aabb2[4] - aabb2[1]) * (aabb2[5] - aabb2[2])
    # Compute the IoU
    iou = inter_vol / (vol_aabb1 + vol_aabb2 - inter_vol)
    return iou

def bbox_iou_3d_batch(aabb1: np.ndarray, aabb2: np.ndarray):
    '''
    Compute the pair-wise IoU of two set of axis-aligned 3D bounding boxes.
    
    Args:
        aabb1: (N, 6) array, [x1, y1, z1, x2, y2, z2] of the first set of bounding boxes
        aabb2: (M, 6) array, [x1, y1, z1, x2, y2, z2] of the second set of bounding boxes
        
    Returns:
        ious: (N, M) array, pair-wise IoU
    '''
    N = aabb1.shape[0]
    M = aabb2.shape[0]
    
    # Expand dimensions to prepare for broadcasting
    aabb1 = np.expand_dims(aabb1, axis=1)  # shape (N, 1, 6)
    aabb2 = np.expand_dims(aabb2, axis=0)  # shape (1, M, 6)
    
    # Compute the coordinates of the intersections
    x1 = np.maximum(aabb1[..., 0], aabb2[..., 0])
    y1 = np.maximum(aabb1[..., 1], aabb2[..., 1])
    z1 = np.maximum(aabb1[..., 2], aabb2[..., 2])
    x2 = np.minimum(aabb1[..., 3], aabb2[..., 3])
    y2 = np.minimum(aabb1[..., 4], aabb2[..., 4])
    z2 = np.minimum(aabb1[..., 5], aabb2[..., 5])
    
    # Compute the dimensions of the intersections
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    d = np.maximum(0, z2 - z1)
    
    # Compute the volume of the intersections
    inter_vol = w * h * d
    
    # Compute the volume of each bounding box
    vol_aabb1 = (aabb1[..., 3] - aabb1[..., 0]) * (aabb1[..., 4] - aabb1[..., 1]) * (aabb1[..., 5] - aabb1[..., 2])
    vol_aabb2 = (aabb2[..., 3] - aabb2[..., 0]) * (aabb2[..., 4] - aabb2[..., 1]) * (aabb2[..., 5] - aabb2[..., 2])
    
    # Compute the IoUs
    ious = inter_vol / (vol_aabb1 + vol_aabb2 - inter_vol)
    
    return ious

@dataclass
class EvalQuery3dbox:
    ckpt_folder: Path
    source_path: Optional[str] = None
    
    threshold_mode: str = "fixed"
    threshold_value: float = 0.6
    
    debug_wandb: bool = False
    
    query_type: str = "2davg" # "3dpoint" or "2davg"
    n_query_samples: int = 5 # Number of query features to sample (only for crossview query_type)

    eval_on_gg: bool = False # Evaluation using the Gaussian Grouping model
    gg_ckpt_folder: Optional[str] = None # Path to the pretrained Guassian Grouping model
    
    def check_args(self):
        assert self.threshold_mode in ["fixed", "gt"], f"Invalid threshold_mode: {self.threshold_mode}"
        assert self.query_type in ["3dpoint", "2davg"], f"Invalid query_type: {self.query_type}"
        
    def main(self) -> None:
        self.check_args()
        
        model, scene, cfg = load_from_model_path(self.ckpt_folder, source_path=self.source_path)
        save_root = self.ckpt_folder

        if self.eval_on_gg:
            from model.gaussian_grouping import GaussianGrouping
            assert self.gg_ckpt_folder is not None, "gg_ckpt_folder must be specified"
            print(f"Loading GaussainGrouping model from {self.gg_ckpt_folder}")
            cfg.model.name = "gaussian_grouping"
            cfg.model.gg_ckpt_folder = self.gg_ckpt_folder
            model = GaussianGrouping(cfg, scene)

            save_root = Path(self.gg_ckpt_folder)
            
        gaussians = model.gaussians
        
        xyz_scene = gaussians.get_xyz.detach().cpu().numpy() # (N, 3)
        xyz_scene_homo = np.concatenate((xyz_scene, np.ones((xyz_scene.shape[0], 1))), axis=1)

        if self.eval_on_gg:
            features = gaussians.get_objects.squeeze(1).detach().cpu().numpy() # (N, D)
        else:
            features = gaussians.get_features_extra.detach().cpu().numpy() # (N, D)
            
        tree = KDTree(xyz_scene)
        
        # Load the queries to be evaluated
        query_path = Path(cfg.scene.source_path) / "3dbox_query.json"
        query_all = json.load(open(query_path, "r"))
        instance_ids = list(query_all.keys())

        # Load and compute query features from training images
        precomputed_features = None
        if self.query_type == "2davg":
            # Load the sampled query
            sampled_query_path = Path(cfg.scene.source_path) / "2dseg_query_sample.json"
            with open(sampled_query_path, "r") as f:
                sampled_query = json.load(f)
            dataloader = scene.get_data_loader("train", shuffle=False, num_workers=8)
            # dataloader = scene.get_data_loader("train", shuffle=False, num_workers=8, limit=200)
            print(len(dataloader))
            precomputed_features = compute_sampled_features(dataloader, model, sampled_query, self.n_query_samples)
            
        
        if self.debug_wandb:
            wandb.init(
                entity="surreal_gs",
                project="debug"
            )

        eval_logs = []
        seg_dynamic_ids = set(scene.scene_info.seg_dynamic_ids)
        seg_static_ids = set(scene.scene_info.seg_static_ids)
        for seg_id in tqdm(instance_ids):
            if int(seg_id) in seg_dynamic_ids:
                seg_type = "dynamic"
            elif int(seg_id) in seg_static_ids:
                seg_type = "static"
            else:
                raise ValueError(f'seg_id {seg_id} not found in either seg_static_ids or seg_dynamic_ids!')
            
            query = query_all[seg_id]
            
            gt_aabb_object = np.array(query['gt_aabb_object']) # (6, )
            gt_T_scene_object = np.array(query['gt_T_scene_object']) # (4, 4)
            
            # Convert the aabb from xxyyzz to xyzxyz
            gt_aabb_object = gt_aabb_object[[0, 2, 4, 1, 3, 5]] # (6, )
            
            if self.query_type == "3dpoint":
                # Search for the nearest Gaussian point to the sampled query location
                sample_xyz_scene = np.array(query['sample_xyz_scene']) # (3, )
                sample_xyz_object = np.array(query['sample_xyz_object']) # (3, )
                D, I = tree.query(sample_xyz_scene, k = 1)
                query_feature = features[I] # (D, )
            elif self.query_type == "2davg":
                if seg_id not in precomputed_features:
                    continue
                query_feature = precomputed_features[seg_id]
                query_feature = query_feature.detach().cpu().numpy() # (C, )
            
            feature_dists = np.linalg.norm(features - query_feature[None], axis=-1) # (N, )
            
            # Compute the Gaussian positions in object frame
            xyz_object_homo = (np.linalg.inv(gt_T_scene_object) @ xyz_scene_homo.T).T # (N, 4)
            xyz_object = xyz_object_homo[:, :3] # (N, 3)
            
            if self.debug_wandb:
                pcds = []
                pcd_all_object = np.concatenate([
                    xyz_object, 
                    np.asarray([128, 128, 128])[None].repeat(xyz_object.shape[0], axis=0)
                ], axis=1)
                pcds.append(pcd_all_object)

                pcd_sample_object = np.concatenate([
                    sample_xyz_object[None], 
                    np.asarray([255, 0, 0])[None]
                ], axis=1)
                pcds.append(pcd_sample_object)

                pcd = np.concatenate(pcds, axis=0)
                pcd[:, 0] *= -1

                wandb.log({
                    f"gaussian_vis/xyz_object_{seg_id}": wandb.Object3D({
                        "type": "lidar/beta",
                        "points": pcd
                    })
                }, commit=True)

            
            if self.threshold_mode == "fixed":
                select_idx = (feature_dists < self.threshold_value).nonzero()
                select_xyz_object = xyz_object[select_idx]  # (M, 3)
                select_aabb_object = np.concatenate((select_xyz_object.min(axis=0), select_xyz_object.max(axis=0)), axis=0) # (6, )
                threshold = self.threshold_value
                iou = bbox_iou_3d(gt_aabb_object, select_aabb_object)
            elif self.threshold_mode == "gt":
                dist_max = feature_dists.max().item()
                thresholds = np.linspace(0, dist_max, 300) # (300, )
                select_idx = (feature_dists[None] < thresholds[:, None]) # (300, N)
                # print(xyz_object.shape, select_idx.shape)

                select_aabb_object = []
                for idx, th in enumerate(thresholds):
                    select_xyz_object = xyz_object[select_idx[idx]] # (M, 3)
                    if len(select_xyz_object) == 0:
                        select_aabb_object.append(np.zeros((6,)))
                    else:
                        select_aabb_object.append(np.concatenate((
                            select_xyz_object.min(axis=0), # (3, )
                            select_xyz_object.max(axis=0) # (3, )
                        ), axis=0)) # (6, )
                select_aabb_object = np.stack(select_aabb_object, axis=0) # (300, 6)
                
                # print(select_aabb_object)
                
                ious = bbox_iou_3d_batch(select_aabb_object, gt_aabb_object) # (300, 1)
                best_idx = np.argmax(ious)
                threshold = thresholds[best_idx]
                iou = ious[best_idx, 0]
            else:
                raise ValueError(f"Unknown threshold_mode: {self.threshold_mode}")
                
            
            eval_logs.append({
                "seg_type": seg_type, 
                "seg_id": seg_id, 
                "iou": iou, 
                "threshold": threshold
            })
            
        df_eval_logs = pd.DataFrame(eval_logs)
        if self.query_type == "3dpoint":
            save_folder_name = "3dbox_eval"
        elif self.query_type == "2davg":
            save_folder_name = "3dbox_eval_cross"
        save_path = save_root / save_folder_name / f"test_logs_{self.threshold_mode}_{self.threshold_value}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_eval_logs.to_csv(save_path, index=False)
        
        static_miou = df_eval_logs[df_eval_logs['seg_type'] == 'static']['iou'].mean()
        dynamic_miou = df_eval_logs[df_eval_logs['seg_type'] == 'dynamic']['iou'].mean()
        
        print(f"static mIoU: {static_miou}, dynamic mIoU: {dynamic_miou}")
        




if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(EvalQuery3dbox).main()
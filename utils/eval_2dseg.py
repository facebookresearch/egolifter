# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from pathlib import Path
from glob import glob
from natsort import natsorted

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import torch
import torch.nn.functional as F
import tyro

from model import get_model
from model.vanilla import VanillaGaussian
from scene import Scene
from utils.general_utils import to_scalar

def mask_iou(gt, pred):
    '''
    Compute the IoU between two masks
    
    Args:
        gt: torch.Tensor, shape (H, W), dtype bool
        pred: torch.Tensor, shape (H', W'), dtype bool
    
    Returns:
        IoU: float
    '''
    if gt.shape != pred.shape:
        # Resize pred to gt
        pred = F.interpolate(
            pred.unsqueeze(0).unsqueeze(0).float(), 
            size=gt.shape, mode='bilinear'
        ).squeeze(0).squeeze(0)
        
        pred = (pred > 0.5).bool()
        
    assert gt.shape == pred.shape
    assert gt.dtype == torch.bool
    
    intersection = (gt & pred).sum()
    union = (gt | pred).sum()
    return intersection / union

def mask_iou_batch(gt, pred):
    '''
    Compute the IoU between two set of masks
    
    Args:
        gt: torch.Tensor, shape (B, H, W) or (1, H, W), dtype bool
        pred: torch.Tensor, shape (B, H', W'), dtype bool
        
    Returns:
        IoU: torch.Tensor, shape (B, )
    '''
    assert gt.ndim == pred.ndim == 3
    assert gt.dtype == torch.bool
    
    if gt.shape != pred.shape:
        # Resize pred to gt
        pred = F.interpolate(
            pred.unsqueeze(0).float(), 
            size=gt.shape[-2:], mode='bilinear'
        ).squeeze(0)
        pred = (pred > 0.5).bool()
        
    assert gt.shape[1:] == pred.shape[1:]
        
    intersection = (gt & pred).sum(dim=(-2, -1))
    union = (gt | pred).sum(dim=(-2, -1))
    
    return intersection / union

@torch.no_grad()
def compute_sampled_features(dataloader, model, sampled_query, n_query_samples) -> dict[str, torch.Tensor]:
    inst_to_feat = {}
    
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image_name = batch['image_name'][0]
        if image_name not in sampled_query:
            continue
        
        query_this_image = sampled_query[image_name]
        viewpoint_cam = dataloader.dataset.get_camera(batch['idx'].item())
        render_pkg = model(viewpoint_cam, render_feature=True)
        render_features = render_pkg['render_features'].permute(1, 2, 0) # (H, W, C)
        
        for inst_id, query in query_this_image.items():
            query_point_u_rel = query['query_point_rel'][0]
            query_point_v_rel = query['query_point_rel'][1]
            query_feature = render_features[
                round(query_point_u_rel * render_features.shape[0]), 
                round(query_point_v_rel * render_features.shape[1])
            ] # (C, )
            
            if inst_id not in inst_to_feat:
                inst_to_feat[inst_id] = []
            inst_to_feat[inst_id].append(query_feature.cpu())
    
    for inst_id in inst_to_feat:
        inst_to_feat[inst_id] = torch.stack(inst_to_feat[inst_id], dim=0)[:n_query_samples]
        inst_to_feat[inst_id] = inst_to_feat[inst_id].mean(dim=0) # (C, )
    
    return inst_to_feat


@torch.no_grad()
def eval_query_2dseg(
    scene: Scene,
    dataloader: torch.utils.data.DataLoader,
    model: VanillaGaussian,
    threshold_mode: str = 'fixed',
    threshold_value: float = 0.6,
    query_type: str = "inview",
    precomputed_features: dict[str, torch.Tensor] = None
) -> tuple[float, pd.DataFrame]:
    assert threshold_mode in ['fixed', 'gt'], f"Invalid threshold_mode: {threshold_mode}"
    assert query_type in ['inview', 'crossview'], f"Invalid query_type: {query_type}"
    
    if query_type == "inview":
        # The query features are sampled on each corresponding image
        pass
    else:
        # The query features are sampled on some random images
        assert precomputed_features is not None, "precomputed_features must be provided for crossview query_type"
    
    eval_logs = []
    
    seg_dynamic_ids = set(scene.scene_info.seg_dynamic_ids)
    seg_static_ids = set(scene.scene_info.seg_static_ids)
    
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        query_2dseg = batch['query_2dseg']
        gt_seg_mask = batch['seg_mask'][0].squeeze(0) # (H, W)
        gt_seg_mask = gt_seg_mask.cuda()
        viewpoint_cam = dataloader.dataset.get_camera(batch['idx'].item())
        
        # Get the mask for valid pixels
        valid_mask = scene.valid_mask_by_name[viewpoint_cam.valid_mask_subpath][1.0] # (1, H, W)
        valid_mask = valid_mask.squeeze(0) # (H, W)
        valid_mask = (valid_mask > 0.5)
        
        gt_seg_mask = gt_seg_mask * valid_mask.long() # (H, W)
        
        render_pkg = model(viewpoint_cam, render_feature=True)
        render_features = render_pkg['render_features'].permute(1, 2, 0) # (H, W, C)
    
        for seg_id_str in query_2dseg:
            seg_id = int(seg_id_str)
            if seg_id in seg_dynamic_ids:
                seg_type = "dynamic"
            elif seg_id in seg_static_ids:
                seg_type = "static"
            else:
                raise ValueError(f'seg_id {seg_id} not found in either seg_static_ids or seg_dynamic_ids!')
            
            gt_seg_this = gt_seg_mask == to_scalar(seg_id)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(image_vis)
            # plt.imshow(gt_seg_this.cpu().numpy(), alpha=0.5)
            # plt.show()

            if query_type == "inview":
                query = query_2dseg[seg_id_str]
                query_point_u_rel = query['query_point_rel'][0].item()
                query_point_v_rel = query['query_point_rel'][1].item()
                query_feature = render_features[
                    round(query_point_u_rel * render_features.shape[0]), 
                    round(query_point_v_rel * render_features.shape[1])
                ] # (C, )
            else:
                if seg_id_str not in precomputed_features:
                    continue
                query_feature = precomputed_features[seg_id_str]
                query_feature = query_feature.to(render_features) # (C, )
            
            # image_vis = batch['image'].squeeze(0).permute(1, 2, 0).numpy().copy()
            # image_vis = (image_vis * 255).astype('uint8')
            # query_point_u = int(query_point_u_rel * image_vis.shape[0])
            # query_point_v = int(query_point_v_rel * image_vis.shape[1])
            # cv2.circle(image_vis, (query_point_v, query_point_u), 5, (0, 0, 255), -1)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(image_vis)
            # plt.show()
            
            # Compute the distance between query feature and all other features
            dists = torch.norm(render_features - query_feature, dim=-1) # (H, W)

            if threshold_mode == 'fixed':
                threshold = threshold_value
                pred = dists < threshold
                pred = pred & valid_mask # (H, W)
                iou = mask_iou(gt_seg_this, pred)
            elif threshold_mode == 'gt':
                # Get the similarity threshold by maximizing the IoU w.r.t GT
                thresholds = np.linspace(0, 3, 300) # (300, )
                thresholds = torch.from_numpy(thresholds).float().cuda()
                pred = dists.unsqueeze(0) < thresholds[:, None, None] # (300, H, W)
                pred = pred & valid_mask[None] # (300, H, W)
                ious = mask_iou_batch(gt_seg_this[None], pred) # (300, )
                idx = ious.argmax()
                threshold = thresholds[idx].item()
                pred = pred[idx]
                iou = ious[idx]

            # dists_vis = dists.cpu().numpy()
            # dists_vis = np.clip(dists_vis, 0, 1)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(image_vis)
            # # plt.imshow(dists_vis, alpha=0.5, cmap='jet')
            # plt.imshow(pred.cpu().numpy(), alpha=0.5)
            # plt.colorbar()
            # plt.show()
            
            eval_logs.append({
                "batch_idx": batch_idx,
                "image_name": viewpoint_cam.image_name,
                "seg_type": seg_type,
                "seg_id": to_scalar(seg_id),
                "iou": iou.item(),
                "threshold": to_scalar(threshold),
            })
                
    df_eval_logs = pd.DataFrame(eval_logs)
    static_miou = df_eval_logs[df_eval_logs['seg_type'] == 'static']['iou'].mean()
    dynamic_miou = df_eval_logs[df_eval_logs['seg_type'] == 'dynamic']['iou'].mean()
    return static_miou, dynamic_miou, df_eval_logs
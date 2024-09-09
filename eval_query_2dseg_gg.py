'''
This file should be run using the gaussian_grouping environment
'''

import argparse
import os
import cv2 # Needed for CCDB
import json

from typing import Optional
from dataclasses import dataclass

from pathlib import Path
import pandas as pd
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from scene import Scene

from utils.eval_2dseg import mask_iou, mask_iou_batch
from utils.routines import load_from_model_path
from utils.general_utils import to_scalar
from utils.gaussian_grouping import parse_namespace


from gaussian_grouping.scene import Scene as gg_Scene
from gaussian_grouping.gaussian_renderer import GaussianModel as gg_GaussianModel
from gaussian_grouping.gaussian_renderer import render as gg_render


@dataclass
class EvalQuery2dseg:
    ckpt_folder: Path
    gg_ckpt_folder: str
    
    source_path: Optional[str] = None

    threshold_mode: str = "fixed"
    threshold_value: float = 0.6
    
    query_type: str = "inview" # "inview" or "crossview"
    n_query_samples: int = 5 # Number of query features to sample (only for crossview query_type)

    def check_args(self):
        assert self.threshold_mode in ["fixed", "gt"], f"Invalid threshold_mode: {self.threshold_mode}"
        assert self.query_type in ["inview", "crossview"], f"Invalid query_type: {self.query_type}"
        

    @torch.no_grad()
    def main(self) -> None:
        self.check_args()

        '''
        Load our own model
        '''
        _, scene, cfg = load_from_model_path(self.ckpt_folder, source_path=self.source_path)
        
        if self.source_path is None:
            self.source_path = cfg.scene.source_path
        
        '''
        Load the GaussianGrouping Model
        '''
        ckpt_args_path = os.path.join(self.gg_ckpt_folder, "cfg_args")
        # Load the dataset arguments and overwrite the model path and source path
        dataset_args = argparse.Namespace(**parse_namespace(ckpt_args_path))
        dataset_args.model_path = self.gg_ckpt_folder
        dataset_args.source_path = cfg.scene.source_path
        
        # Construct the pipeline arguments
        pipeline_args = argparse.Namespace(
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
            self.gg_ckpt_folder, "point_cloud","iteration_"+str(gg_scene.loaded_iter),"classifier.pth"))
        )
        bg_color = [1,1,1] if dataset_args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        
        precomputed_features = None
        # Compute the features for each instance from a sampled set of queries on training images
        if self.query_type == "crossview":
            # Load the sampled query
            sampled_query_path = Path(cfg.scene.source_path) / "2dseg_query_sample.json"
            with open(sampled_query_path, "r") as f:
                sampled_query = json.load(f)
            dataloader = scene.get_data_loader("train", shuffle=False, num_workers=8)
            # dataloader = scene.get_data_loader("train", shuffle=False, num_workers=8, limit=200)
            # precomputed_features = compute_sampled_features(dataloader, model, sampled_query, self.n_query_samples)

            '''
            The loop to compute the precomputed features
            '''
            precomputed_features = {}
    
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                image_name = batch['image_name'][0]
                if image_name not in sampled_query:
                    continue
                
                query_this_image = sampled_query[image_name]
                viewpoint_cam = dataloader.dataset.get_camera(batch['idx'].item())

                # render_pkg = model(viewpoint_cam, render_feature=True)
                # render_features = render_pkg['render_features'].permute(1, 2, 0) # (H, W, C)
                
                render_pkg = gg_render(viewpoint_cam, gaussians, pipeline_args, background)
                render_features = render_pkg["render_object"].permute(1, 2, 0) # (H, W, C)
                
                for inst_id, query in query_this_image.items():
                    query_point_u_rel = query['query_point_rel'][0]
                    query_point_v_rel = query['query_point_rel'][1]
                    query_feature = render_features[
                        round(query_point_u_rel * render_features.shape[0]), 
                        round(query_point_v_rel * render_features.shape[1])
                    ] # (C, )
                    
                    if inst_id not in precomputed_features:
                        precomputed_features[inst_id] = []
                    precomputed_features[inst_id].append(query_feature.cpu())
            
            for inst_id in precomputed_features:
                precomputed_features[inst_id] = torch.stack(precomputed_features[inst_id], dim=0)[:self.n_query_samples]
                precomputed_features[inst_id] = precomputed_features[inst_id].mean(dim=0) # (C, )
                
        
        for subset in ['valid', "test"]:
            print(f"Evaluating subset: {subset} ...")
            dataloader = scene.get_data_loader(subset, shuffle=False, num_workers=0, limit=200)
            
            '''
            The evaluation loop
            '''
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
                
                # render_pkg = model(viewpoint_cam, render_feature=True)
                # render_features = render_pkg['render_features'].permute(1, 2, 0) # (H, W, C)

                render_pkg = gg_render(viewpoint_cam, gaussians, pipeline_args, background)
                render_features = render_pkg["render_object"].permute(1, 2, 0) # (H, W, C)
            
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

                    if self.query_type == "inview":
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
                    dist_max = dists.max().item()

                    if self.threshold_mode == 'fixed':
                        threshold = self.threshold_value
                        pred = dists < threshold
                        pred = pred & valid_mask # (H, W)
                        iou = mask_iou(gt_seg_this, pred)
                    elif self.threshold_mode == 'gt':
                        # Get the similarity threshold by maximizing the IoU w.r.t GT
                        thresholds = np.linspace(0, dist_max, 300) # (300, )
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
            
            print(f"{subset}: static mIoU: {static_miou}, dynamic mIoU: {dynamic_miou}")
            
            # Save the evaluation logs to the ckpt_folder
            if self.query_type == "inview":
                save_path = Path(self.gg_ckpt_folder) / "2dseg_eval" / f"{subset}_logs_{self.threshold_mode}_{self.threshold_value}.csv"
            elif self.query_type == "crossview":
                save_path = Path(self.gg_ckpt_folder) / "2dseg_eval_cross" / f"{subset}_logs_{self.threshold_mode}_{self.threshold_value}_{self.n_query_samples}.csv"
            else:
                raise ValueError(f"Invalid query_type: {self.query_type}")
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_eval_logs.to_csv(save_path, index=False)
    
    
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(EvalQuery2dseg).main()
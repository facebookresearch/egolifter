# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


'''
Run SAM segmentation on the 2D query images. 
'''
from dataclasses import dataclass

from pathlib import Path
from glob import glob
from typing import Optional
from natsort import natsorted

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import torch
import torch.nn.functional as F
import tyro
from segment_anything import sam_model_registry, SamPredictor

from model import get_model
from model.vanilla import VanillaGaussian
from scene import Scene
from utils.general_utils import to_scalar
from utils.eval_2dseg import mask_iou_batch
from utils.routines import load_from_model_path

@torch.no_grad()
def eval_query_2dseg_sam(
    scene: Scene,
    dataloader: torch.utils.data.DataLoader,
    model: VanillaGaussian,
    predictor: SamPredictor,
    selection_mode: str = "gt",
):
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
        
        render_pkg = model(viewpoint_cam, render_feature=False)
        
        render_image = render_pkg['render'] # (3, H, W)
        
        # Expects an image in HWC uint8 format, with pixel values in [0, 255].
        image = render_image.permute(1, 2, 0) # (H, W, 3)
        image = image.cpu().numpy() # (H, W, 3)
        image = (image * 255).round().astype(np.uint8) # (H, W, 3)
        image = np.rot90(image, k=-1) # (W, H, 3)

        predictor.set_image(image)
        
        for seg_id_str in query_2dseg:
            seg_id = int(seg_id_str)
            if seg_id in seg_dynamic_ids:
                seg_type = "dynamic"
            elif seg_id in seg_static_ids:
                seg_type = "static"
            else:
                raise ValueError(f'seg_id {seg_id} not found in either seg_static_ids or seg_dynamic_ids!')
            
            gt_seg_this = gt_seg_mask == to_scalar(seg_id)

            query = query_2dseg[seg_id_str]
            query_point_u_rel = query['query_point_rel'][0].item()
            query_point_v_rel = query['query_point_rel'][1].item()
            query_point_x = round(query_point_v_rel * image.shape[1])
            query_point_y = round(query_point_u_rel * image.shape[0])
            
            # Get the point after clockwise rotation by 90 degrees
            query_point_x_rot = image.shape[1] - query_point_y
            query_point_y_rot = query_point_x
            
            input_point = np.array([[query_point_x_rot, query_point_y_rot]])
            input_label = np.array([1])
            
            masks, conf, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            ) # (N, W, H)
            
            # # Visualization for sanity check 
            # plt.imshow(image)
            # plt.show()
            # plt.imshow(np.rot90(gt_seg_this.cpu().numpy(), k=-1))
            # plt.show()            
            # for i in range(len(masks)):
            #     plt.imshow(masks[i])
            #     plt.show()
            
            # Rotate the masks back to the original orientation
            masks = np.rot90(masks, k=1, axes=(1, 2)) # (N, H, W)
            masks = masks & valid_mask[None].cpu().numpy() # (N, H, W)
            
            # plt.imshow(gt_seg_this.cpu().numpy())
            # plt.show() 
            # for i in range(len(masks)):
            #     plt.imshow(masks[i])
            #     plt.show()
            
            masks = torch.from_numpy(masks).cuda()
            ious = mask_iou_batch(gt_seg_this[None], masks) # (N, )
            if selection_mode == "gt":
                idx = ious.argmax()
                pred = masks[idx]
                iou = ious[idx]
            elif selection_mode == "conf":
                idx = conf.argmax()
                pred = masks[idx]
                iou = ious[idx]
                
            eval_logs.append({
                "batch_idx": batch_idx,
                "image_name": viewpoint_cam.image_name,
                "seg_type": seg_type,
                "seg_id": to_scalar(seg_id),
                "iou": to_scalar(iou),
                "threshold": 0.0,
            })
                
    df_eval_logs = pd.DataFrame(eval_logs)
    static_miou = df_eval_logs[df_eval_logs['seg_type'] == 'static']['iou'].mean()
    dynamic_miou = df_eval_logs[df_eval_logs['seg_type'] == 'dynamic']['iou'].mean()
    return static_miou, dynamic_miou, df_eval_logs



@dataclass
class EvalQuery2dseg:
    ckpt_folder: Path # For convenient loading of a configuration
    source_path: Optional[str] = None
    
    sam_model_type: str = "vit_h"
    sam_ckpt_path: str = "./Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
    selection_mode: str = "gt"

    def main(self) -> None:
        model, scene, cfg = load_from_model_path(self.ckpt_folder, source_path=self.source_path)
        
        # Initialize the SAM model
        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_ckpt_path)
        sam = sam.eval().cuda()
        predictor = SamPredictor(sam)
        
        for subset in ['valid', "test"]:
            print(f"Evaluating subset: {subset} ...")
            dataloader = scene.get_data_loader(subset, shuffle=False, num_workers=0, limit=200)

            static_miou, dynamic_miou, df_eval_logs = eval_query_2dseg_sam(
                scene, dataloader, model, predictor, self.selection_mode
            )
            print(f"{subset}: static mIoU: {static_miou}, dynamic mIoU: {dynamic_miou}")
            
            # Save the evaluation logs to the ckpt_folder
            save_path = self.ckpt_folder / "2dseg_eval_sam" / f"{subset}_logs_{self.selection_mode}.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_eval_logs.to_csv(save_path, index=False)
    
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(EvalQuery2dseg).main()
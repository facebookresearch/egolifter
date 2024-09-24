# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

from typing import Optional
from dataclasses import dataclass

from pathlib import Path

import torch
import torch.nn.functional as F
import tyro

from model import get_model
from scene import Scene

from utils.eval_2dseg import eval_query_2dseg, compute_sampled_features
from utils.routines import load_from_model_path

@dataclass
class EvalQuery2dseg:
    ckpt_folder: Path
    source_path: Optional[str] = None

    threshold_mode: str = "fixed"
    threshold_value: float = 0.6
    
    query_type: str = "inview" # "inview" or "crossview"
    n_query_samples: int = 5 # Number of query features to sample (only for crossview query_type)
    

    def check_args(self):
        assert self.threshold_mode in ["fixed", "gt"], f"Invalid threshold_mode: {self.threshold_mode}"
        assert self.query_type in ["inview", "crossview"], f"Invalid query_type: {self.query_type}"
        

    def main(self) -> None:
        self.check_args()
        model, scene, cfg = load_from_model_path(self.ckpt_folder, source_path=self.source_path)
        
        precomputed_features = None
        # Compute the features for each instance from a sampled set of queries on training images
        if self.query_type == "crossview":
            # Load the sampled query
            sampled_query_path = Path(cfg.scene.source_path) / "2dseg_query_sample.json"
            with open(sampled_query_path, "r") as f:
                sampled_query = json.load(f)
            dataloader = scene.get_data_loader("train", shuffle=False, num_workers=8)
            # dataloader = scene.get_data_loader("train", shuffle=False, num_workers=8, limit=200)
            precomputed_features = compute_sampled_features(dataloader, model, sampled_query, self.n_query_samples)
        
        for subset in ['valid', "test"]:
            print(f"Evaluating subset: {subset} ...")
            dataloader = scene.get_data_loader(subset, shuffle=False, num_workers=0, limit=200)
            static_miou, dynamic_miou, df_eval_logs = eval_query_2dseg(
                scene, dataloader, model, self.threshold_mode, self.threshold_value, 
                self.query_type, precomputed_features
            )
            print(f"{subset}: static mIoU: {static_miou}, dynamic mIoU: {dynamic_miou}")
            
            # Save the evaluation logs to the ckpt_folder
            if self.query_type == "inview":
                save_path = self.ckpt_folder / "2dseg_eval" / f"{subset}_logs_{self.threshold_mode}_{self.threshold_value}.csv"
            elif self.query_type == "crossview":
                save_path = self.ckpt_folder / "2dseg_eval_cross" / f"{subset}_logs_{self.threshold_mode}_{self.threshold_value}_{self.n_query_samples}.csv"
            else:
                raise ValueError(f"Invalid query_type: {self.query_type}")
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_eval_logs.to_csv(save_path, index=False)
    
    
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(EvalQuery2dseg).main()
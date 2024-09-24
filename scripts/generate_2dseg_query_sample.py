# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Sample a set of queries for cross-view 2D segmentation evaluation
Ensure that all queries are sampled from the training images. 
'''

import os, sys

import json
from pathlib import Path
import gzip, pickle
from PIL import Image

import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import scipy
import cv2

from dataclasses import dataclass
import tyro


@dataclass
class Sample2dsegQuery:
    data_root: Path
    scene_name: str
    
    def main(self) -> None:
        random.seed(42)
        
        input_folder = self.data_root / self.scene_name
        
        # First, load the metadata and get the idx for training images
        metadata_path = input_folder / "transforms.json"
        with open(metadata_path) as json_file:
            frames = json.loads(json_file.read())['frames']
            
        frames = [f for f in frames if f["camera_name"] == "rgb"]
        unique_devices = set()
        for idx, frame in enumerate(frames):
            unique_devices.add(frame["device"])
        
        # Split the cameras into seen and novel views
        all_idx = np.arange(0, len(frames))
        if len(unique_devices) == 1:
            seen_idx = all_idx[:int(0.8*len(all_idx))]
            novel_idx = all_idx[int(0.8*len(all_idx)):]
        else:
            seen_idx = np.asarray([idx for idx in all_idx if frames[idx]["device"] == 0])
            novel_idx = np.asarray([idx for idx in all_idx if frames[idx]["device"] != 0])

        # Validation frames within the seen views
        valid_idx = seen_idx[::5]
        if len(valid_idx) > 2000:
            valid_idx = valid_idx[np.linspace(0, len(valid_idx)-1, 2000, dtype=int)]

        # The remaining seen frames are the training frames
        train_idx = np.setdiff1d(seen_idx, valid_idx)
        train_image_subpaths = [frames[idx]["image_path"] for idx in train_idx]
        train_image_subpaths = set(train_image_subpaths)
        print("Number of training images:", len(train_image_subpaths))
        
        # Now sample the queries
        all_query_path = input_folder / "2dseg_query.json"
        sampled_query_path = input_folder / "2dseg_query_sample.json"
                
        with open(all_query_path, "r") as f:
            all_queries = json.load(f)
            
        # Iterate over all the queries and sort them by instance id
        inst_to_queries = {}
        for image_subpath in all_queries.keys():
            # Only use the queries from training images
            if image_subpath not in train_image_subpaths:
                continue
            
            querys = all_queries[image_subpath]
            for inst_id in querys.keys():
                if inst_id not in inst_to_queries:
                    inst_to_queries[inst_id] = []

                q = querys[inst_id]
                q['image_subpath'] = image_subpath
                inst_to_queries[inst_id].append(q)
        
        # Sample a subset of queries for each instance
        sampled_query = {} # Still indexed by image_subpath, then by instance id
        for inst_id in inst_to_queries.keys():
            queries = inst_to_queries[inst_id]
            sampled_queries = random.sample(queries, min(len(queries), 10))
            for q in sampled_queries:
                image_subpath = q['image_subpath']
                if image_subpath not in sampled_query:
                    sampled_query[image_subpath] = {}
                sampled_query[image_subpath][inst_id] = q
        
        with open(sampled_query_path, "w") as f:
            json.dump(sampled_query, f, indent=4)
            
        print(f"Sampled queries saved to {sampled_query_path}")
            
    
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Sample2dsegQuery).main()
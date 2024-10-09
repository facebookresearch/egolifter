# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import glob
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

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinSkeletonProvider,
   AriaDigitalTwinDataPathsProvider,
   bbox3d_to_line_coordinates,
   bbox2d_to_image_coordinates,
   MotionType, 
   utils as adt_utils,
)

from scipy.stats import truncnorm

# https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


@dataclass
class Generate3dboxQuery:
    raw_root: Path
    data_root: Path
    scene_name: str

    seed: int = 42
    
    def main(self) -> None:
        # Filter out the instances such that each instance should be reasonably visible
        # (queried for 2D instance segmentation) in at least one frame
        processed_folder = self.data_root / self.scene_name
        query_2dseg_path = processed_folder / "2dseg_query.json"
        query_2dseg = json.load(open(query_2dseg_path, "r"))
        instance_ids_queried = set()
        for image_subpath, frame_query in query_2dseg.items():
            for instance_id, instance_query in frame_query.items():
                instance_ids_queried.add(int(instance_id))
        
        # Load the object pose information from the raw data folder
        # Handle the new data format, where each device recording has its sequence. 
        raw_folder = self.raw_root / self.scene_name
        subseq_paths = sorted(glob.glob(str(raw_folder) + "_*"))
        assert len(subseq_paths) > 0, f"No subsequence found in {raw_folder}"
        paths_provider = AriaDigitalTwinDataPathsProvider(subseq_paths[0])
        
        data_paths = paths_provider.get_datapaths(skeleton_flag=False)
        gt_provider = AriaDigitalTwinDataProvider(data_paths)
        
        stream_id = StreamId("214-1")
        
        img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)

        # Get the 3D bbox at the first timestamp (we only care about static object for now)
        timestamp_ns = img_timestamps_ns[0]
        boundingbox_3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(timestamp_ns)

        assert boundingbox_3d_with_dt.is_valid()
        dt_ns = abs(boundingbox_3d_with_dt.dt_ns())
        boundingbox_3d = boundingbox_3d_with_dt.data()
        
        # Cut off at 2 * std
        X = get_truncated_normal(mean=0, sd=0.25, low=0, upp=1)

        query_3dbox = {}
        
        # FIx the seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        for instance_id, bbox in boundingbox_3d.items():
            if instance_id not in instance_ids_queried:
                continue

            T_scene_object = bbox.transform_scene_object.to_matrix()
            aabb_object = bbox.aabb
            xyz_min_object = aabb_object[[0, 2, 4]]
            xyz_max_object = aabb_object[[1, 3, 5]]

            # sample a 3D points as query
            sample = X.rvs(3)
            sample_xyz_object = xyz_min_object + sample * (xyz_max_object - xyz_min_object)
            
            # Transform the 3D point to the scene coordinate
            sample_xyz_scene = T_scene_object @ np.hstack([sample_xyz_object, 1.0])
            sample_xyz_scene = sample_xyz_scene[:3]
            
            assert instance_id not in query_3dbox, f"instance_id {instance_id} already exists in query_3dbox"
            
            query_3dbox[instance_id] = {
                "sample_xyz_scene": sample_xyz_scene.tolist(),
                "sample_xyz_object": sample_xyz_object.tolist(),
                "gt_aabb_object": aabb_object.tolist(),
                "gt_T_scene_object": T_scene_object.tolist(),
            }

        # Save the query to the data_root
        query_save_path = processed_folder / "3dbox_query.json"
        with open(query_save_path, "w") as f:
            json.dump(query_3dbox, f, indent=4)

    
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Generate3dboxQuery).main()
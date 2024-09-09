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
class Generate2dsegQuery:
    data_root: Path
    scene_name: str
    
    mask_area_threshold: float = 0.001
    seed: int = 42

    debug: bool = False
    
    def main(self) -> None:
        input_folder = self.data_root / self.scene_name
        
        # Load the meta information from processed ADT data folder
        metadata_path = input_folder / "transforms.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        instances_meta_path = input_folder / "instances.json"
        with open(instances_meta_path, "r") as f:
            instances_metadata = json.load(f)
            
        frames = metadata['frames']
        
        query_all = {}
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        if self.debug:
            frames = frames[:10]
        
        for frame in tqdm(frames):
            segmentation_path = input_folder / frame['segmentation_path']
            segmentation_viz_path = input_folder / frame['segmentation_viz_path']
            
            with gzip.open(segmentation_path, 'rb') as f:
                segmentation = pickle.load(f)
            
            segmentation_viz = np.asarray(Image.open(segmentation_viz_path))
            
            instance_ids, instance_areas = np.unique(segmentation, return_counts=True)
            sort_idx = np.argsort(instance_areas)[::-1]
            instance_ids = instance_ids[sort_idx]
            instance_areas = instance_areas[sort_idx]
            
            image_area = segmentation.size
            
            # all_mask = np.zeros_like(segmentation, dtype=bool)
            # all_mask_eroded = np.zeros_like(segmentation, dtype=bool)
            # mask_with_query = np.zeros_like(segmentation_viz)
            
            query_frame = {}
            
            for inst_id, inst_area in zip(instance_ids, instance_areas):
                if inst_id == 0: # outside of valid mask
                    continue

                if inst_area < self.mask_area_threshold * image_area:
                    continue
                
                inst_name = instances_metadata[str(inst_id)]['instance_name']
                
                if inst_name == "ApartmentEnv": # The background (walls, floor, ceiling)
                    continue
                
                inst_mask = segmentation == inst_id

                # binary erosion the mask
                inst_mask_eroded = scipy.ndimage.binary_erosion(inst_mask, iterations=5)
                
                if inst_mask_eroded.sum() < 10:
                    continue
                
                # Randomly select a point as mask query
                mask_indices = np.argwhere(inst_mask_eroded)
                query_point = mask_indices[np.random.choice(len(mask_indices))]
                
                query_point_rel = query_point / np.array(inst_mask.shape)
                query_frame[int(inst_id)] = {
                    "query_point": query_point.tolist(),
                    "query_point_rel": query_point_rel.tolist(),
                }

                # all_mask = np.logical_or(all_mask, inst_mask)
                # all_mask_eroded = np.logical_or(all_mask_eroded, inst_mask_eroded)

                # # Draw a circle around the query point
                # mask_with_query[inst_mask_eroded] = 255
                # query_point_cv = tuple(query_point[::-1])
                # cv2.circle(mask_with_query, query_point_cv, 5, (255, 0, 0), -1)
                
            query_all[frame['image_path']] = query_frame

        query_save_path = input_folder / "2dseg_query.json"
        with open(query_save_path, "w") as f:
            json.dump(query_all, f, indent=4)


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Generate2dsegQuery).main()
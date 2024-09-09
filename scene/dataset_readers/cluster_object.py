import gzip
import pickle
from PIL import Image
from typing import Any, NamedTuple, Optional
import cv2

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scene.cameras import Camera

from .base import SceneInfo, SceneType, getNerfppNorm, storePly, fetchPly
from .aria import readAriaSceneInfo

def readClusterObjectInfo(
        input_folder:str, 
        cluster_folder:str, 
        cluster_id:int,
        white_background:bool,
        min_iou:float,
    ):
    input_folder = Path(input_folder)
    cluster_folder = Path(cluster_folder)
    
    assert input_folder.exists(), f"Could not find input folder at {input_folder}"
    assert cluster_folder.exists(), f"Could not find cluster folder at {cluster_folder}"
    
    # Get the scene info of the original training sequence
    scene_info = readAriaSceneInfo(input_folder, camera_used="rgb", stride=1)

    # Get the clustering results
    point_labels_instance_path = cluster_folder / "point_labels_instance.npy"
    point_labels_instance = np.load(str(point_labels_instance_path))
    matching_results_path = cluster_folder / "mask_cluster_matching.csv"
    df_all = pd.read_csv(str(matching_results_path))

    df = df_all[
        (df_all["cluster_id"] == cluster_id) &
        (df_all["inter_over_union"] > min_iou)
    ]
    if len(df) == 0:
        raise ValueError(f"No matching results for cluster {cluster_id}. Please retry")
    
    print(f"Found {len(df)} matching results for cluster {cluster_id}")
        
    df = df.sort_values(by="inter_over_union", ascending=False).reset_index(drop=True)

    # Load vignette image, which is needed to mask out the white background
    vignette_path = input_folder / "rgb_vignette.png"
    vignette = Image.open(str(vignette_path))
    vignette = np.asarray(vignette)
    
    cam_list_cluster = []
    for row in df.itertuples():
        view_id = row.view_id
        image_name = row.image_name
        
        cam_info = scene_info.train_cameras[view_id]
        assert cam_info.image_name == image_name, f"{cam_info.image_name} from training camera does not match {image_name} from matching results"
        
        # assert isinstance(cam_info.image, Image.Image), f"Camera image is not an Image.Image, but {type(cam_info.image)}"
        # image = np.asarray(cam_info.image).copy()

        det_path = input_folder / "gsa_det_none_sam" / f"{image_name}.pkl.gz"
        with gzip.open(str(det_path), 'rb') as f:
            det_result = pickle.load(f)
        
        mask = det_result["mask"][row.mask_id]
        mask = cv2.resize(
            mask.astype(np.float32), 
            (cam_info.width, cam_info.height),
            interpolation=cv2.INTER_LINEAR
        )
        if scene_info.scene_type == SceneType.ARIA:
            mask = np.rot90(mask, k=1, axes=(0, 1)).copy()
        mask = (mask > 0.5).astype(np.uint8)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # if white_background:
        #     # image[~mask] = np.ones_like(image[~mask]) * 255
        #     # image = np.bitwise_or(image, np.bitwise_not(mask))
            
        #     # Apply the vignette image on the background
        #     image = image * mask + (1 - mask) * vignette
        # else:
        #     # image[~mask] = np.zeros_like(image[~mask])
        #     mask = mask * 255
        #     image = np.bitwise_and(image, mask)
            
        # image = Image.fromarray(image)
        
        cam_cluster = Camera(
            colmap_id=view_id,
            uid=view_id,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FoVx,
            FoVy=cam_info.FoVy,
            image_width=cam_info.image_width,
            image_height=cam_info.image_height,
            image_name=image_name,
            image_path=cam_info.image_path,
            scale=1.0,
            camera_name='rgb',
            fid=None,
        )
        
        cam_list_cluster.append(cam_cluster)
        
    nerf_normalization = getNerfppNorm(cam_list_cluster)

    scene_info_cluster = SceneInfo(
        point_cloud=scene_info.point_cloud,
        train_cameras=cam_list_cluster,
        test_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=scene_info.ply_path,
        scene_type=SceneType.CLUSTER,
        valid_mask_by_name=scene_info.valid_mask_by_name,
        vignette_by_name=scene_info.vignette_by_name,
    )

    return scene_info_cluster
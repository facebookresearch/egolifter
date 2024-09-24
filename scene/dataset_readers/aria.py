# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import os
import warnings
from PIL import Image
import pandas as pd

from utils.graphics_utils import BasicPointCloud, getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from scene.cameras import Camera

from .base import SceneInfo, SceneType, getNerfppNorm, storePly, fetchPly
from .mps_point_cloud_reader import read_global_point_cloud

def read_aria_ply(points_path):
    # read pointcloud
    # import projectaria_tools.core.mps as mps
    # points = mps.read_global_point_cloud(str(points_path))

    # projectaria_tools is not available on the cluster
    # So use our own re-implementation
    points = read_global_point_cloud(str(points_path))

    # filter the point cloud by inverse depth and depth
    filtered_points = []
    for point in points:
        if (point.inverse_distance_std < 0.001 and point.distance_std < 0.15):
            filtered_points.append(point)

    # example: get position of this point in the world coordinate frame
    points_world = []
    for point in filtered_points:
        position_world = point.position_world
        points_world.append(position_world)

    xyz = np.stack(points_world, axis=0)
    rgb = (np.ones_like(points_world) * 128).astype(np.uint8) # Initialize the color of all points to gray
    # ply_path = str(input_folder / "points_splat.ply")
    ply_path = tempfile.NamedTemporaryFile(suffix=".ply", delete=False).name
    storePly(ply_path, xyz, rgb)
    
    return ply_path

def load_aria_frames(
        metadata_paths: list[str],
        camera_used: str,
        stride: int,
    ):

    # Go through the transforms and create the camera infos
    frames = []
    for metadata_path in metadata_paths:
        with open(metadata_path) as json_file:
            metadata = json.loads(json_file.read())
        frames += metadata["frames"]

    if camera_used == "rgb":
        frames = [f for f in frames if f["camera_name"] == "rgb"]
    elif camera_used == "rgblseg":
        frames = [f for f in frames if f["camera_name"] in ["rgb", "rgb-lseg-32"]]
    elif camera_used == "rgblseg24":
        frames = [f for f in frames if f["camera_name"] in ["rgb", "rgb-lseg-24"]]
    elif camera_used == "rgbdinov2s":
        frames = [f for f in frames if f["camera_name"] in ["rgb", "rgb-dinov2s-32"]]
    elif camera_used == "rgbdinos8":
        frames = [f for f in frames if f["camera_name"] in ["rgb", "rgb-dinos8-32"]]
    elif camera_used == "rgbdinov25s":
        frames = [f for f in frames if f["camera_name"] in ["rgb", "rgb-dinov25s-32"]]
    elif camera_used == "rgbdinov25s_11_token":
        frames = [f for f in frames if f["camera_name"] in ["rgb", "rgb-dinov25s_11_token-32"]]
    elif camera_used == "slam":
        frames = [f for f in frames if f["camera_name"] in ["slaml", "slamr"]]
    elif camera_used == "rgbslam":
        frames = [f for f in frames if f["camera_name"] in ["rgb", "slaml", "slamr"]]
    else:
        raise NotImplementedError(f"Camera name {camera_used} not supported (yet).")
    
    # This sorts the frame list by first camera_name, then by capture time. 
    frames.sort(key=lambda f: f["image_path"])
    
    if stride != 1:
        frames = frames[::stride]
    
    return frames

# def readAriaSceneInfo(
#         input_folder:str, 
#         camera_used:str = "rgb", 
#         stride:int = 1,
#         scene_name: str = "none",
#     ):

def readAriaSceneInfo(
        cfg_scene,
    ):
    
    input_folder = cfg_scene.source_path
    camera_used = cfg_scene.camera_name
    stride = cfg_scene.stride
    scene_name = cfg_scene.scene_name
    
    input_folder = Path(input_folder)
    
    # Load the metadata for all the frames
    metadata_paths = [str(input_folder / "transforms.json")]
    frames = load_aria_frames(metadata_paths, camera_used, stride = stride)

    # Get a normalizer for the timestamp
    timestamps = np.asarray([f["timestamp"] for f in frames])
    ts_min, ts_max = np.min(timestamps), np.max(timestamps)
    timestamp_to_fid = lambda ts: (ts - ts_min + 0.0) / (ts_max - ts_min) # Normalize the timestamp to [0, 1]

    # Load the vignette and valid masks
    camera_names = set([f["camera_name"] for f in frames])
    print(f"Using cameras: {camera_names}")
    
    # The valid mask and vignette are indices by their subpath
    valid_mask_by_name, vignette_by_name = {}, {}

    # Iterate and create the Camera objects
    all_cam_list = []
    unique_devices = set()
    for idx, frame in enumerate(frames):
        image_path_full = str(input_folder / frame["image_path"])
        unique_devices.add(frame["device"])

        transform_matrix = np.asarray(frame["transform_matrix"])

        fx, fy, cx, cy = frame["fx"], frame["fy"], frame["cx"], frame["cy"]
        width, height = frame["w"], frame["h"]
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        transform_matrix = np.linalg.inv(transform_matrix)
        R = transform_matrix[:3,:3]
        R = R.T # row-major vs. col-major
        t = transform_matrix[:3,3]
        
        cam_args = dict(
            colmap_id=idx,
            uid=idx,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image_width=width,
            image_height=height,
            image_name=frame["image_path"],
            image_path=image_path_full,
            scale=1.0,
            camera_name=frame["camera_name"],
            fid=timestamp_to_fid(frame["timestamp"]),
            scene_name=scene_name,
            exposure=frame["exposure_duration_s"],  # Not used
            gain=frame['gain'],   # Not used
            valid_mask_subpath=frame["mask_subpath"],
            vignette_subpath=frame["vignette_subpath"],
            scene_folder=input_folder,
        )
        
        if "segmentation_path" in frame and frame['segmentation_path']:
            cam_args["seg_path"] = str(input_folder / frame["segmentation_path"])
            
        if frame["mask_subpath"] not in valid_mask_by_name:
            valid_mask_path = input_folder / frame["mask_subpath"]
            if valid_mask_path.exists():
                valid_mask_by_name[frame["mask_subpath"]] = Image.open(str(valid_mask_path))

        if frame["vignette_subpath"] not in vignette_by_name:
            vignette_path = input_folder / frame["vignette_subpath"]
            if valid_mask_path.exists():
                vignette_by_name[frame["vignette_subpath"]] = Image.open(str(vignette_path))
        
        cam = Camera(**cam_args)

        all_cam_list.append(cam)
        
    all_idx = np.arange(0, len(all_cam_list))

    # Split the cameras into seen and novel views
    if cfg_scene.no_novel: # All frames are in seen subset
        seen_idx = all_idx
        novel_idx = np.array([])
    else:
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

    if cfg_scene.all_seen_train: # All seen frames are in the training set
        train_idx = seen_idx
    else: # The remaining seen frames are the training frames
        train_idx = np.setdiff1d(seen_idx, valid_idx)

    # Validation frames in the novel views
    valid_novel_idx = novel_idx[::2]
    if len(valid_novel_idx) > 2000:
        valid_novel_idx = valid_novel_idx[np.linspace(0, len(valid_novel_idx)-1, 2000, dtype=int)]
    
    # testing frams are the rest of the novel views
    test_idx = np.setdiff1d(novel_idx, valid_novel_idx)
    
    train_camera_infos = [all_cam_list[i] for i in train_idx]
    valid_camera_infos = [all_cam_list[i] for i in valid_idx]
    valid_novel_camera_infos = [all_cam_list[i] for i in valid_novel_idx]
    test_camera_infos = [all_cam_list[i] for i in test_idx]
    
    # Load the initial point cloud from MPS
    points_path = input_folder / "global_points.csv.gz"
    if points_path.exists():
        ply_path = read_aria_ply(points_path)
        point_cloud = fetchPly(ply_path)
    else:
        warnings.warn(f"Could not find the point cloud at {points_path}. Using an empty point cloud.")
        ply_path = None
        point_cloud = BasicPointCloud(
            np.zeros((0, 3), dtype=np.float32), 
            np.zeros((0, 3), dtype=np.float32), 
            np.zeros((0, 3), dtype=np.float32), 
        )
    
    scene_info_args = dict(
        point_cloud=point_cloud,
        train_cameras=train_camera_infos,
        valid_cameras=valid_camera_infos,
        test_cameras=test_camera_infos,
        nerf_normalization=getNerfppNorm(train_camera_infos),
        ply_path=ply_path,
        scene_type=SceneType.ARIA,
        valid_novel_cameras=valid_novel_camera_infos,
        valid_mask_by_name=valid_mask_by_name,
        vignette_by_name=vignette_by_name,
    )
    
    metadata = json.loads(open(metadata_paths[0], "r").read())
    if "skeleton_instances_ids" in metadata:
        print("Loading the semantic segmentation info")
        static_instances_ids = metadata["static_instances_ids"]
        dynamic_instances_ids = metadata["dynamic_instances_idx"]
        
        # Filter out the objects that move less than 2cm
        scene_objects_path = input_folder / "scene_objects.csv"
        df_scene_objects = pd.read_csv(scene_objects_path)
        true_dynamic_instances_ids = []
        for object_uid in dynamic_instances_ids:
            df_instance = df_scene_objects[df_scene_objects["object_uid"] == object_uid]
            if len(df_instance) == 0:
                # This should be a skeleton instance, considered as dynamic
                true_dynamic_instances_ids.append(object_uid)
                continue
            
            instance_xyz = df_instance[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].to_numpy() # (T, 3)
            instance_xyz_range = np.max(instance_xyz, axis=0) - np.min(instance_xyz, axis=0)
            instance_move_range = np.linalg.norm(instance_xyz_range)
            if instance_move_range > 0.02:
                true_dynamic_instances_ids.append(object_uid)
            else:
                static_instances_ids.append(object_uid)
        
        dynamic_instances_ids = true_dynamic_instances_ids
        scene_info_args["seg_static_ids"] = static_instances_ids
        scene_info_args["seg_dynamic_ids"] = dynamic_instances_ids

    query_2dseg_path = input_folder / "2dseg_query.json"
    if query_2dseg_path.exists():
        print("Loading the 2D segmentation info")
        query_2dseg = json.loads(open(query_2dseg_path, "r").read())
        scene_info_args["query_2dseg"] = query_2dseg

    # Create the scene info object
    scene_info = SceneInfo(**scene_info_args)
    
    if ply_path is not None:
        os.remove(ply_path)

    return scene_info
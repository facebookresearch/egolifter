import glob
import math
import os

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from pathlib import Path
from scene.cameras import Camera
from natsort import natsorted

from .base import SceneInfo, SceneType, getNerfppNorm, storePly, fetchPly

def get_replica_intrisic(img_h, img_w):
    fx = 600.0
    fy = 600.0
    cx = 599.5
    cy = 339.5
    return fx, fy, cx, cy

def readReplicaInfo(input_folder: str, image_stride:int = 1, pcd_stride:int = 1):
    input_folder = Path(input_folder)

    traj_path = input_folder / "traj.txt"
    points_path = input_folder / "rgb_cloud" / "pointcloud.ply"
    rgb_paths = glob.glob(str(input_folder / "results" / "frame*.jpg"))
    depth_paths = glob.glob(str(input_folder / "results" / "depth*.png"))

    rgb_paths = natsorted(rgb_paths)
    depth_paths = natsorted(depth_paths)

    assert len(rgb_paths) > 0, "No RGB images found at {}".format(str(input_folder / "results" / "frame*.jpg"))
    assert len(rgb_paths) == len(depth_paths), "Number of RGB and depth images must match"
    assert os.path.exists(points_path), "Could not find pointcloud at {}".format(points_path)
    assert os.path.exists(traj_path), "Could not find camera trajectory at {}".format(traj_path)

    # Read point cloud
    pointcloud = fetchPly(points_path, stride=pcd_stride)

    # Load the poses
    poses = np.loadtxt(traj_path, delimiter=" ").reshape(-1, 4, 4)
    assert len(poses) == len(rgb_paths), f"Number of poses ({len(poses)}) and number of images ({len(rgb_paths)}) must match"

    # Load the intrinsics
    img_h, img_w = 680, 1200
    fx, fy, cx, cy = get_replica_intrisic(img_h, img_w)
    fovx = focal2fov(fx, img_w)
    fovy = focal2fov(fy, img_h)


    # Load the images
    cam_list = []
    for idx in range(0, len(rgb_paths), image_stride):
        rgb_path = rgb_paths[idx]
        depth_path = depth_paths[idx]
        pose = poses[idx]

        pose = np.linalg.inv(pose)
        R = pose[:3,:3]
        R = R.T
        t = pose[:3,3]

        cam = Camera(
            colmap_id=idx,
            uid=idx,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image_width=img_w,
            image_height=img_h,
            image_name=rgb_path.split("/")[-1].split(".")[0],
            image_path=rgb_path,
            scale=1.0,
            camera_name='rgb',
            fid=0.0,
            scene_folder=input_folder,
        )
        cam_list.append(cam)

    # # Split the train/valid sets according to that used in panoptic lifting
    # all_idx = np.arange(0, len(cam_list))
    # valid_idx = np.arange(0, len(cam_list), 4) # 25% of the images are used for validation
    # train_idx = np.setdiff1d(all_idx, valid_idx)
    
    # Split the train/valid sets according to OmniSeg3D
    train_idx = np.arange(0, 2000, 20)
    valid_idx = np.arange(10, 2000, 80)
    
    train_camera_infos = [cam_list[idx] for idx in train_idx]
    valid_camera_infos = [cam_list[idx] for idx in valid_idx]
    test_camera_infos = []
    
    # print("Train:", [cam.image_name for cam in train_camera_infos])
    # print()
    # print("Valid:", [cam.image_name for cam in valid_camera_infos])

    nerf_normalization = getNerfppNorm(train_camera_infos)

    scene_info = SceneInfo(
        point_cloud=pointcloud,
        train_cameras=train_camera_infos,
        valid_cameras=valid_camera_infos,
        test_cameras=test_camera_infos,
        nerf_normalization=nerf_normalization,
        ply_path=str(points_path),
        scene_type=SceneType.REPLICA_SEMANTIC,
        valid_mask_by_name={},
        vignette_by_name={},
    )

    return scene_info
import os
from PIL import Image
from typing import Any
from dataclasses import dataclass

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.cameras import Camera

from .base import SceneInfo, SceneType, getNerfppNorm, storePly, fetchPly

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_list = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam = Camera(
                colmap_id=idx,
                uid=idx,
                R=R,
                T=T,
                FoVx=FovX,
                FoVy=FovY,
                image_width=image.size[0],
                image_height=image.size[1],
                image_name=image_name,
                image_path=image_path,
                scale=1.0,
                camera_name='rgb',
                fid=frame_time,
            )

            cam_list.append(cam)
            
    return cam_list

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_list = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_list = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_list.extend(test_cam_list)
        test_cam_list = []

    nerf_normalization = getNerfppNorm(train_cam_list)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_list,
                           valid_cameras=[],
                           test_cameras=test_cam_list,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           scene_type=SceneType.NERF_SYNTHETIC,
                           valid_mask_by_name=None,
                           vignette_by_name=None,
                           )
    return scene_info
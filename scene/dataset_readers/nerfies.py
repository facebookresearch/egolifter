# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from PIL import Image

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from utils.sh_utils import SH2RGB
from utils.camera_utils import camera_nerfies_from_JSON
from scene.gaussian_model import BasicPointCloud
from scene.cameras import Camera

from .base import SceneInfo, SceneType, getNerfppNorm, storePly, fetchPly


def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        all_img = train_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_list = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])

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
            fid=fid,
        )
        
        cam_list.append(cam)

    sys.stdout.write('\n')
    return cam_list, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           valid_cameras=[],
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           scene_type=SceneType.NERFIES,
                           valid_mask_by_name=None,
                           vignette_by_name=None,
                           )
    return scene_info
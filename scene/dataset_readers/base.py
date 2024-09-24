# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image
from plyfile import PlyData, PlyElement
import numpy as np

from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scene.cameras import Camera

class SceneType(Enum):
    COLMAP = 1
    NERF_SYNTHETIC = 2
    ARIA = 3
    REPLICA_SEMANTIC = 4
    CLUSTER = 5
    NERFIES = 6

@dataclass
class SceneInfo:
    point_cloud: BasicPointCloud
    train_cameras: list[Camera]
    valid_cameras: list[Camera]
    test_cameras: list[Camera]
    nerf_normalization: dict
    ply_path: str
    scene_type: SceneType
    valid_novel_cameras: list[Camera] = field(default_factory=list)
    valid_mask_by_name: Optional[dict] = field(default_factory=dict)
    vignette_by_name: Optional[dict] = field(default_factory=dict)
    seg_static_ids: Optional[np.ndarray] = None
    seg_dynamic_ids: Optional[np.ndarray] = None
    query_2dseg: Optional[dict] = None

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path, stride:int=1):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    if stride > 1:
        print(f"Subsampling point cloud of length {positions.shape[0]} with stride {stride}...")
        positions = positions[::stride]
        colors = colors[::stride]
        normals = normals[::stride]
        print(f"New point cloud size: {positions.shape[0]}")
    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals = None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    
def load_image_or_feat(image_path: str) -> Image.Image | np.ndarray:
    if image_path.endswith(".npy"):
        feat = np.load(image_path)  # Must be in (H, W, C)
        # TODO: remove this normalization on other features
        feat = feat / np.linalg.norm(feat, axis=2, keepdims=True)
        if np.isnan(feat).sum() > 0:
            print(f"ERROR: NaNs in feature {image_path}")
            import pdb; pdb.set_trace()
        return feat
    elif image_path.endswith(".jpg") or image_path.endswith(".png"):
        image = Image.open(image_path)
        if image.mode == "L":
            image = image.convert("RGB")

        return image
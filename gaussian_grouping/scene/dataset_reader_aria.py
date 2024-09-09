import glob
import os
import sys
import warnings
from PIL import Image
from typing import NamedTuple
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement

from gaussian_grouping.utils.sh_utils import SH2RGB
from gaussian_grouping.scene.gaussian_model import BasicPointCloud
from gaussian_grouping.scene.utils import fetchPly, storePly, CameraInfo, SceneInfo, getNerfppNorm
from gaussian_grouping.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


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

def readAriaSceneInfo(input_folder:str, camera_used:str = "rgb", stride:int = 1):
    input_folder = Path(input_folder)

    # transform_path = input_folder / "transforms.json"
    transform_paths = glob.glob(str(input_folder / "transforms.json"))
    points_path = input_folder / "global_points.csv.gz"

    # Extra files for Gaussian Grouping
    object_dir = 'deva_object_mask'
    objects_folder = str(input_folder / object_dir)

    # # read pointcloud
    # import projectaria_tools.core.mps as mps
    # points = mps.read_global_point_cloud(str(points_path), mps.StreamCompressionMode.GZIP)

    # projectaria_tools is not available on the cluster
    # So use our own re-implementation
    from .mps_point_cloud_reader import read_global_point_cloud
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
    ply_path = str(input_folder / "points_splat.ply")
    storePly(ply_path, xyz, rgb)

    # Go through the transforms and create the camera infos
    frames = []
    for transform_path in transform_paths:
        with open(transform_path) as json_file:
            transformes = json.loads(json_file.read())
        frames += transformes["frames"]

    if camera_used == "rgb":
        frames = [f for f in frames if f["camera_name"] == "rgb"]
    else:
        raise NotImplementedError(f"Camera name {camera_used} not supported (yet).")

    camera_names = set([f["camera_name"] for f in frames])
    print(f"Using cameras: {camera_names}")

    # Load the valid mask for each camera type
    valid_mask_by_name = {}
    for camera_name in camera_names:
        valid_mask_path = str(input_folder / f"{camera_name}_mask.png")
        assert os.path.exists(valid_mask_path), f"Could not find valid mask for camera {camera_name} at {valid_mask_path}"
        valid_mask = Image.open(valid_mask_path)
        valid_mask_by_name[camera_name] = valid_mask
    
    # Load the vignette for each camera type
    vignette_by_name = {}
    for camera_name in camera_names:
        vignette_path = input_folder / f"{camera_name}_vignette.png"
        # assert vignette_path.exists(), f"Could not find vignette for camera {camera_name} at {vignette_path}"
        if vignette_path.exists():
            vignette = Image.open(str(vignette_path))
            vignette_by_name[camera_name] = vignette
        else:
            warnings.warn(f"Could not find vignette for camera {camera_name} at {vignette_path}. Skipping...")
            vignette_by_name[camera_name] = None


    # This sorts the frame list by first camera_name, then by capture time. 
    frames.sort(key=lambda f: f["image_path"])
    
    total_frames = len(frames)
    frames = frames[::stride]
    print(f"Using {len(frames)} out of {total_frames} frames. (stride={stride})")

    cam_infos = []
    for idx, frame in enumerate(frames):
        image_path_full = str(input_folder / frame["image_path"])

        # image = load_image_or_feat(image_path_full)
        
        transform_matrix = np.asarray(frame["transform_matrix"])

        fx, fy, cx, cy = frame["fx"], frame["fy"], frame["cx"], frame["cy"]
        width, height = frame["w"], frame["h"]
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        transform_matrix = np.linalg.inv(transform_matrix) # Why?
        R = transform_matrix[:3,:3]
        R = R.T # Why?
        t = transform_matrix[:3,3]

        image_name = image_path_full.split("/")[-1].split(".")[0]
        object_path = os.path.join(objects_folder, image_name + '.png')
        objects = Image.open(object_path) if os.path.exists(object_path) else None

        if frame['device'] == 0:
            assert objects is not None, f"Could not find object mask for image {image_name} at {object_path}"

        if frame['device'] > 0:
            image_name = f"dev1_{image_name}"
        
        cam_info = CameraInfo(
            uid = idx,
            R = R,
            T = t,
            FovX = fovx,
            FovY = fovy,
            image = None,
            image_path = image_path_full,
            image_name = image_name,
            width = width,
            height = height,
            objects = objects,
            device=frame['device'],
        )

        cam_infos.append(cam_info)
        
    # Ensure GG is using the same training set as our own codebase
    unique_devices = [0]
    if (input_folder / "dev1_images").exists():
        unique_devices.append(1)
        print("Found dev1_images, assuming two devices.")
    
    all_idx = np.arange(0, len(cam_infos))
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

    train_idx = np.setdiff1d(seen_idx, valid_idx)

    train_camera_infos = [cam_infos[i] for i in train_idx]
    test_camera_infos = [cam_infos[i] for i in valid_idx]
    
    print(f"Found {len(train_camera_infos)} training cameras and {len(test_camera_infos)} validation cameras.")

    nerf_normalization = getNerfppNorm(train_camera_infos)

    scene_info = SceneInfo(
        point_cloud=fetchPly(ply_path),
        train_cameras=train_camera_infos,
        test_cameras=test_camera_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        valid_mask_by_name=valid_mask_by_name,
        vignette_by_name=vignette_by_name,
    )

    return scene_info

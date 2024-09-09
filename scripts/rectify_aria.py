import argparse
import csv
import gzip
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import cv2
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from pprint import pprint

import imageio.v2 as imageio
import open3d as o3d

from projectaria_tools.core import calibration
from projectaria_tools.core.sophus import SE3
import projectaria_tools.core.mps as mps
import torch

def better_camera_frustum(camera_pose, img_h, img_w, scale=3.0, color=[0, 0, 1]):
    # Convert camera pose tensor to numpy array
    if isinstance(camera_pose, torch.Tensor):
        camera_pose = camera_pose.numpy()
    
    # Define near and far distance (adjust these as needed)
    near = scale * 0.1
    far = scale * 1.0
    
    # Define frustum dimensions at the near plane (replace with appropriate values)
    frustum_h = near
    frustum_w = frustum_h * img_w / img_h  # Set frustum width based on its height and the image aspect ratio
    
    # Compute the 8 points that define the frustum
    points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                u = x * (frustum_w // 2 if z == -1 else frustum_w * far / near)
                v = y * (frustum_h // 2 if z == -1 else frustum_h * far / near)
                d = near if z == -1 else far # Negate depth here
                # d = -near if z == -1 else -far # Negate depth here
                point = np.array([u, v, d, 1]).reshape(-1, 1)
                transformed_point = (camera_pose @ point).ravel()[:3]
                # transformed_point[0] *= -1  # Flip X-coordinate
                points.append(transformed_point) # Using camera pose directly
                # points.append((camera_pose_np @ point).ravel()[:3]) # Using camera pose directly
    
    # Create lines that connect the 8 points
    lines = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [3, 7], [2, 6]]
    
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return frustum

CAMERA_NAMES = ["rgb", "slaml", "slamr"]

@dataclass
class AriaFrame:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_params: np.ndarray
    w: int
    h: int
    file_path: str
    transform_matrix: np.ndarray
    timestamp: int
    exposure_duration_s: float
    gain: float
    camera_name: str

def read_frames_from_metadata(input_folder: Path, image_folder_name:str = "images_raw"):
    metadata_path = input_folder / "transforms_raw.json"
    with open(metadata_path, 'r')  as f:
        metadata = json.load(f)

    camera_model = metadata['camera_model']
    frames_raw = metadata['frames']

    frames = []
    for f in frames_raw:
        file_path = input_folder / image_folder_name / f['file_path'].split("/")[-1]

        # If the matched time difference is larger than 3_000_000ns (3ms), we skip this frame
        if f['time_diff_ns'] > 3_000_000:
            continue
        
        # get intrinsic calibration parameters
        # The fisheye624 calibration parameter vector has the following interpretation:
        # params = [f c_u c_v k_0 k_1 k_2 k_3 k_4 k_5 p_0 p_1 s_0 s_1 s_2 s_3]
        # where f is the focal length, (c_u, c_v) is the principal point, k_i are the six
        # radial distortion coefficients, p_i are the two tangential distortion coefficients,
        # and s_i are the four thin prism coefficients.
        frames.append(AriaFrame(
            fx = f['fl_x'],
            fy = f['fl_y'],
            cx = f['cx'],
            cy = f['cy'],
            distortion_params = np.asarray(f['distortion_params']),
            w = f['w'],
            h = f['h'],
            file_path = str(file_path),
            transform_matrix = np.asarray(f['transform_matrix']),
            timestamp=f['timestamp'],
            exposure_duration_s=f['exposure_duration_s'],
            gain=f['gain'],
            camera_name=f['camera_name'],
        ))

    print(f"There are {len(frames)} valid frames out of {len(frames_raw)} frames in total.")

    # frames_rgb = [f for f in frames if f.camera_name == "rgb"]
    # frames_slam = [f for f in frames if f.camera_name.startswith("slam")]

    return camera_model, frames

def undistort_image(frame: AriaFrame, image_raw: np.ndarray, ow: int, oh: int, f: float):
    # we assume fx and fy are the same
    proj_params = np.asarray([frame.fx, frame.cx, frame.cy])
    proj_params = np.concatenate((proj_params, frame.distortion_params))

    transform_matrix = SE3()

    frame_calib = calibration.CameraCalibration(
        "camera-rgb",
        calibration.CameraModelType.FISHEYE624,
        proj_params,
        transform_matrix,
        frame.w,
        frame.h,
        None,
        90.0,
        "", # Serial number
    )
    pinhole_calib = calibration.get_linear_camera_calibration(ow, oh, f)
    image = calibration.distort_by_calibration(image_raw, pinhole_calib, frame_calib)
    return image
    

# the multiprocessor version of the code
def process_frame(frame: AriaFrame, args: argparse.Namespace):
    output_folder = args.output_root / args.scene_name

    filename = os.path.basename(frame.file_path)

    # Force the output images to use png format
    image_output_subpath = f"images/{filename}"[:-3] + "jpg" 
    image_output_path = output_folder / image_output_subpath

    os.makedirs(os.path.dirname(image_output_path), exist_ok=True)

    image_raw = imageio.imread(frame.file_path)

    if frame.camera_name == "rgb":
        output_width = args.output_rgb_width
        output_height = args.output_rgb_height
        output_focal = args.output_rgb_focal
    elif frame.camera_name in ['slaml', 'slamr']:
        output_width = args.output_slam_width
        output_height = args.output_slam_height
        output_focal = args.output_slam_focal
    else:
        raise ValueError(f"Unknown camera name {frame.camera_name}")

    if args.debug_coord:
        # Debugging: Only care about the open3d visualization
        # Skip the computation
        output_fx = 0
        output_fy = 0
        output_cx = 0
        output_cy = 0
    else:
        image = undistort_image(frame, image_raw, output_width, output_height, output_focal)

        # Reference: https://github.com/facebookresearch/projectaria_tools/blob/main/core/calibration/CameraCalibration.cpp#L153
        output_fx = output_focal
        output_fy = output_focal
        output_cx = (output_width - 1) / 2.0
        output_cy = (output_height - 1) / 2.0

        imageio.imwrite(image_output_path, image)

    # Actually the coordinate system used in COLMAP is 
    # "The local camera coordinate system of an image is defined in a way that the X axis points to the right, 
    # the Y axis to the bottom, and the Z axis to the front as seen from the image, 
    # which is the same as the original Surreal data and different from that in Nerfstudio. 
    # So here are simply changed it back to the original Surreal coordinate system.
    transform = frame.transform_matrix

    return {
        "fx": output_fx,
        "fy": output_fy,
        "cx": output_cx,
        "cy": output_cy,
        "w": output_width,
        "h": output_height,
        "image_path": image_output_subpath,
        "transform_matrix": transform.tolist(),
        "timestamp": frame.timestamp,
        "exposure_duration_s": frame.exposure_duration_s,
        "gain": frame.gain,
        "camera_name": frame.camera_name,
        "segmentation_path": None,
        "segmentation_viz_path": None,
        "device": 0,
        "vignette_subpath": f"{frame.camera_name}_vignette.png",
        "mask_subpath": f"{frame.camera_name}_mask.png",
    }

def get_vignette_by_name(args: argparse.Namespace, frames: list[AriaFrame]) -> dict[str, np.ndarray]:
    input_root = args.input_root
    vignettes = {}

    for n in CAMERA_NAMES:
        frames_n = [f for f in frames if f.camera_name == n]
        if len(frames_n) == 0:
            continue

        input_h = frames_n[0].h
        input_w = frames_n[0].w
        
        if n == "rgb":
            vignette_path = input_root / "vignette_imx577.png"
            if not vignette_path.exists():
                vignette_path = "assets/vignette_imx577.png"
        else:
            vignette_path = input_root / "vignette_ov7251.png"
            if not vignette_path.exists():
                vignette_path = "assets/vignette_ov7251.png"
        
        vignette_raw = imageio.imread(os.path.expanduser(vignette_path))
        vignette_raw = vignette_raw[:, :, :3]
        vignette_raw = cv2.resize(vignette_raw, (input_w, input_h))

        if n == "rgb":
            vignette = undistort_image(frames_n[0], vignette_raw, args.output_rgb_width, args.output_rgb_height, args.output_rgb_focal)
        else:
            vignette = undistort_image(frames_n[0], vignette_raw, args.output_slam_width, args.output_slam_height, args.output_slam_focal)
        
        vignettes[n] = vignette

    return vignettes

def get_mask_by_name(args: argparse.Namespace, frames: list[AriaFrame]) -> dict[str, np.ndarray]:
    masks = {}

    for n in CAMERA_NAMES:
        frames_n = [f for f in frames if f.camera_name == n]
        if len(frames_n) == 0:
            continue
        frame = frames_n[0]

        if n == "rgb":
            output_width = args.output_rgb_width
            output_height = args.output_rgb_height
            output_focal = args.output_rgb_focal
        elif n in ['slaml', 'slamr']:
            output_width = args.output_slam_width
            output_height = args.output_slam_height
            output_focal = args.output_slam_focal
        else:
            raise ValueError(f"Unknown camera name {n}")

        image_raw = imageio.imread(frame.file_path)
        mask_raw = np.zeros(image_raw.shape[:2], dtype=np.uint8)
        radius = int(frame.w * 0.5)
        cv2.circle(mask_raw, (frame.w // 2, frame.h // 2), radius, 255, -1)

        mask = undistort_image(frame, mask_raw, output_width, output_height, output_focal)
        
        masks[n] = mask

    return masks

def main(args: argparse.Namespace):
    input_root = args.input_root
    output_root = args.output_root
    scene_name = args.scene_name

    input_folder = input_root / scene_name
    assert input_folder.exists(), f"Input folder {input_folder} does not exist"
    output_folder = output_root / scene_name

    transform_output_path = output_folder / "transforms.json"
    os.makedirs(output_folder, exist_ok=True)

    input_points_path = input_folder / "global_points.csv.gz"
    if not input_points_path.exists():
        input_points_path = input_folder / "semidense_points.csv.gz"
        
    output_points_path = output_folder / "global_points.csv.gz"
    if not output_points_path.exists():
        shutil.copy(input_points_path, output_points_path)

    # Parse the metadata
    camera_model, frames = read_frames_from_metadata(input_folder)
    
    if args.output_rgb_focal is None:
        frame = next(f for f in frames if f.camera_name == "rgb")
        args.output_rgb_focal = frame.fx
        print(f"No output_rgb_focal provided, using the focal length from raw data: {args.output_rgb_focal}")

    # Read the vignette image, rectify it, and save it
    vignettes = get_vignette_by_name(args, frames)
    for n, vig in vignettes.items():
        output_path = output_folder / f"{n}_vignette.png"
        imageio.imwrite(output_path, vig)

    # Get the mask for each camera, rectify it, and save it
    masks = get_mask_by_name(args, frames)
    for n, mask in masks.items():
        output_path = output_folder / f"{n}_mask.png"
        imageio.imwrite(output_path, mask)

    # Load images and rectify them
    transforms = []
    with multiprocessing.Pool(24) as pool:
        pool_args = [(frame, args) for frame in frames]
        transforms = pool.starmap(process_frame, pool_args)

    if args.debug_coord:
        # A script to verify the coordinate system
        # read pointcloud
        points = mps.read_global_point_cloud(str(input_points_path), mps.StreamCompressionMode.GZIP)

        # filter the point cloud by inverse depth and depth
        filtered_points = []
        for point in points:
            if (point.inverse_distance_std < 0.001 and point.distance_std < 0.15):
                filtered_points.append(point)

        # get position of this point in the world coordinate frame
        points_world = []
        for point in filtered_points:
            position_world = point.position_world
            points_world.append(position_world)

        points_world = np.stack(points_world, axis=0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.paint_uniform_color([0.2, 0.2, 0.2])

        for camera_name in ["rgb", "slaml", "slamr"]:
            geometries = [pcd]
            cmap = plt.get_cmap('jet')
            transforms_cam = [t for t in transforms if t['camera_name'] == camera_name]
            if len(transforms_cam) == 0:
                continue
            
            for i in range(0, len(transforms_cam), 10):
                transf = transforms_cam[i]
                transform = np.asarray(transf['transform_matrix'])
                color = cmap(float(i) / len(transforms_cam))
                frustum = better_camera_frustum(transform, 300, 300, scale=0.15, color=color[:3])
                geometries.append(frustum)

            # Draw a coordinate system at the origin
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            geometries.append(coord_frame)

            o3d.visualization.draw_geometries(geometries, camera_name)

        exit(0)

    with open(transform_output_path, 'w') as f:
        json.dump({
            "camera_model": "pinhole",
            "frames": transforms,
        }, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_root", type=Path, default="/home/qgu/local/data/aria_gs/")
    parser.add_argument("-o", "--output_root", type=Path, default="/tmp/")
    parser.add_argument("-s", "--scene_name", type=str, default="fireplace")

    parser.add_argument("--output_rgb_width", type=int, default=1408)
    parser.add_argument("--output_rgb_height", type=int, default=1408)
    parser.add_argument("--output_rgb_focal", type=int, default=None)

    parser.add_argument("--output_slam_width", type=int, default=480)
    parser.add_argument("--output_slam_height", type=int, default=480)
    parser.add_argument("--output_slam_focal", type=int, default=150)

    parser.add_argument("--mask_image", action="store_true",
                        help="If set, the rectified image will be masked by the valid mask. ")

    parser.add_argument("--start_frame", type=int, default=0, 
                        help="start converting for the n-th frame - The first a few frames tend to have bad exposure.")
    
    parser.add_argument("--debug_coord", action="store_true",
                        help="If set, debugging the coordinate system by visualization. ")
    
    args = parser.parse_args()

    main(args)

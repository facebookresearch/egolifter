# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


'''
Preprocessing script converting the Aria Digital Twin dataset to the 3DGS format.
'''

import gzip
import multiprocessing
from pathlib import Path
import pickle
import shutil
import sys, os, glob
import json, csv
from pprint import pprint
from dataclasses import dataclass
import cv2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import tyro
import imageio


from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.core.sophus import SE3
from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinSkeletonProvider,
   AriaDigitalTwinDataPathsProvider,
   bbox3d_to_line_coordinates,
   bbox2d_to_image_coordinates,
   MotionType, 
   utils as adt_utils,
)

@dataclass
class TimedPoses:
    timestamps: np.ndarray
    translations: np.ndarray
    quaternions: np.ndarray
    
    def get_matrix(self, index: int) -> np.ndarray:
        quat = self.quaternions[index]
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_quat(quat).as_matrix()
        matrix[:3, 3] = self.translations[index]
        return matrix
    
    def get_nearest_index(self, query_timestamp: int) -> int:
        nearest_pose_idx = np.searchsorted(self.timestamps, query_timestamp)
        nearest_pose_idx = np.minimum(nearest_pose_idx, len(self.timestamps) - 1)
        return nearest_pose_idx
    
def crop_and_resize_vignette(
        vignette_img: Image, 
        out_img_size: tuple[int, int]
    ):
    if (
        vignette_img.size[0] == out_img_size[0]
        and vignette_img.size[1] == out_img_size[1]
    ):
        return vignette_img

    assert out_img_size[0] == 1408 and out_img_size[1] == 1408

    offset_x, offset_y = (32, 32)
    w, h = vignette_img.size
    vignette_img = vignette_img.crop((offset_x, offset_y, w - offset_x, h - offset_y))
    vignette_img = vignette_img.resize(out_img_size)
    return vignette_img

def read_trajectory_csv_to_dict(csv_path: str) -> TimedPoses:
    reader = csv.reader(open(csv_path, "r"))
    headers = next(reader)

    try:
        time_and_pose_names = [
            "tracking_timestamp_us",
            "tx_world_device",
            "ty_world_device",
            "tz_world_device",
            "qx_world_device",
            "qy_world_device",
            "qz_world_device",
            "qw_world_device",
        ]
        time_and_pose_columns = [headers.index(name) for name in time_and_pose_names]
    except ValueError:
        print("Could not find world pose columns, trying odometry pose columns")
        time_and_pose_names = [
            "tracking_timestamp_us",
            "tx_odometry_device",
            "ty_odometry_device",
            "tz_odometry_device",
            "qx_odometry_device",
            "qy_odometry_device",
            "qz_odometry_device",
            "qw_odometry_device",
        ]
        time_and_pose_columns = [headers.index(name) for name in time_and_pose_names]
        
    time_and_pose_columns = np.asarray(time_and_pose_columns)
    data = np.array(
        [np.array(row)[time_and_pose_columns] for row in reader], dtype=float
    )

    MICROSEC_TO_NANOSEC = 1e3
    return TimedPoses(
        timestamps=(data[:, 0] * MICROSEC_TO_NANOSEC).astype(int),
        translations=data[:, 1:4],
        quaternions=data[:, 4:8],
    )

    
@dataclass
class ProcessAriaDigitalTwin:
    data_root: Path
    sequence_name: str
    output_root: Path
    
    output_rgb_width: int = 1408
    output_rgb_height: int = 1408
    output_rgb_focal: int = 460
    
    save_raw_images: bool = False

    # output_slam_width: int = 480
    # output_slam_height: int = 480
    # output_slam_focal: int = 150
    
    def main(self) -> None:
        sequence_name = self.sequence_name
        sequence_path = self.data_root / sequence_name
        
        # Output folder
        output_folder = self.output_root / sequence_name
        
        
        # Initialize the data provider for this sequence
        paths_provider = AriaDigitalTwinDataPathsProvider(str(sequence_path))
        all_device_serials = paths_provider.get_device_serial_numbers()

        print("all devices for sequence ", sequence_name, ":")
        for idx, device_serial in enumerate(all_device_serials):
            print("device number - ", idx, ": ", device_serial)
            
        
        output_frame_metadata = []
        for selected_device_number in range(len(all_device_serials)):
            print("processing device number: ", selected_device_number)
            
            data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number, skeleton_flag=True)
            gt_provider = AriaDigitalTwinDataProvider(data_paths)
            
            os.makedirs(output_folder, exist_ok=True)

            # Copy the some files containing metadata of the sequence
            if selected_device_number == 0:
                src_path = data_paths.instances_filepath
                dst_path = output_folder / "instances.json"
                if not dst_path.exists():
                    print("Copying instances.json from {} to {}".format(src_path, dst_path))
                    shutil.copy(src_path, dst_path)

                src_path = data_paths.object_trajectories_filepath
                dst_path = output_folder / "scene_objects.csv"
                if not dst_path.exists():
                    print("Copying scene_objects.csv from {} to {}".format(src_path, dst_path))
                    shutil.copy(src_path, dst_path)

            # Analyze the object instances in this sequence
            instances_ids = gt_provider.get_instance_ids()
            print("there are {} object in this sequence".format(len(instances_ids)))

            skeleton_instances_ids = []
            static_instances_ids = []
            dynamic_instances_idx = []
            for id in instances_ids:
                instance = gt_provider.get_instance_info_by_id(id)
                if "Skeleton" in instance.name:
                    skeleton_instances_ids.append(id)
                if instance.motion_type == MotionType.STATIC:
                    static_instances_ids.append(id)
                elif instance.motion_type == MotionType.DYNAMIC:
                    dynamic_instances_idx.append(id)
                else:
                    print("unknown motion type for instance id: ", id)
                    
            print("there are {} skeleton objects in this sequence".format(len(skeleton_instances_ids)))
            print("there are {} static objects in this sequence".format(len(static_instances_ids)))
            print("there are {} dynamic objects in this sequence".format(len(dynamic_instances_idx)))


            # Load the closed-loop trajectory from MPS files
            data_folder = os.path.dirname(data_paths.aria_vrs_filepath)
            closed_trajectory_csv_path = os.path.join(data_folder, "mps", "slam", "closed_loop_trajectory.csv")
            poses_closed = read_trajectory_csv_to_dict(closed_trajectory_csv_path)

            # Copy the semidense points to the destination folder
            semidense_points_path = os.path.join(data_folder, "mps", "slam", "semidense_points.csv.gz")
            semidense_points_save_path = output_folder / "global_points.csv.gz"
            if selected_device_number > 0:
                semidense_points_save_path = output_folder / f"dev{selected_device_number}_global_points.csv.gz"
            os.system(f"cp {semidense_points_path} {str(semidense_points_save_path)}")

            
            # Get calibration and undistortion information
            stream_id = StreamId("214-1") # The RGB camera
            sensor_name = gt_provider.raw_data_provider_ptr().get_label_from_stream_id(stream_id)
            device_calib = gt_provider.raw_data_provider_ptr().get_device_calibration()
            # TODO: change from factory calibration to online calibration
            src_calib = device_calib.get_camera_calib(sensor_name)
            dst_calib = calibration.get_linear_camera_calibration(
                self.output_rgb_width, 
                self.output_rgb_height, 
                self.output_rgb_focal, 
                sensor_name
            )
            # Reference: https://github.com/facebookresearch/projectaria_tools/blob/main/core/calibration/CameraCalibration.cpp#L153
            output_fx = self.output_rgb_focal
            output_fy = self.output_rgb_focal
            output_cx = (self.output_rgb_width - 1) / 2.0
            output_cy = (self.output_rgb_height - 1) / 2.0
            
            # TODO: change from factory calibration to online calibration
            # TODO: Get the nearest calibration according to the timestamp
            transform_device_cam = gt_provider.get_aria_transform_device_camera(stream_id)


            # Get the valid mask and vignette
            vignette_rgb_path = self.data_root / "vignette_imx577.png"
            vignette_rgb = Image.open(vignette_rgb_path.absolute().as_posix())
            vignette_rgb = crop_and_resize_vignette(vignette_rgb, (self.output_rgb_width, self.output_rgb_height))
            vignette_rgb = np.copy(np.ascontiguousarray(vignette_rgb)[:, :, :3])
            rectified_vignette_rgb = calibration.distort_by_calibration(vignette_rgb, dst_calib, src_calib)

            mask_rgb = np.zeros(rectified_vignette_rgb.shape[:2], dtype=np.uint8)
            radius = int(rectified_vignette_rgb.shape[0] / 2.0)
            cv2.circle(mask_rgb, (self.output_rgb_width // 2, self.output_rgb_height // 2), radius, 255, -1)
            rectified_mask_rgb = calibration.distort_by_calibration(mask_rgb, dst_calib, src_calib)

            vignette_save_subpath = "rgb_vignette.png"
            if selected_device_number > 0:
                vignette_save_subpath = f"dev{selected_device_number}_rgb_vignette.png"
            rectified_vignette_rgb_savepath = output_folder / vignette_save_subpath
            imageio.imwrite(rectified_vignette_rgb_savepath, rectified_vignette_rgb)
            
            mask_save_subpath = "rgb_mask.png"
            if selected_device_number > 0:
                mask_save_subpath = f"dev{selected_device_number}_rgb_mask.png"
            rectified_mask_rgb_savepath = output_folder / mask_save_subpath
            imageio.imwrite(rectified_mask_rgb_savepath, rectified_mask_rgb)
            
            
            # Iterate through all RGB images
            img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)
            print("There are {} frames".format(len(img_timestamps_ns)))

            output_frame_metadata_device = []
            for idx, timestamp_ns in enumerate(tqdm(img_timestamps_ns)):
                image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, stream_id)
                assert image_with_dt.is_valid(), "Image not valid!"
                
                # RGB cameras use rolling shutter, and camera timestamp is the time of the first roll.
                # To better sync with the pose, we approximate timestamp of an RGB at center roll.
                # Therefore offset the timestamp by half-readout time (2.5ms for 1408x1408, 8ms for 2880x2880)
                timestamp_ns_pose_search = timestamp_ns + 2_500_000 # 2.5 ms
                nearest_idx_closed = poses_closed.get_nearest_index(timestamp_ns_pose_search)
                transform_world_device = poses_closed.get_matrix(nearest_idx_closed)
                transform_world_cam = transform_world_device @ transform_device_cam.to_matrix()
                dt_ns_closed = abs(timestamp_ns_pose_search - poses_closed.timestamps[nearest_idx_closed])
                
                # TODO: Change to get_nearest_pose() in Project Aria API
                # TODO: Maybe use interpolation to get better poses
                # Some frames may not have matched closed-loop trajectory
                # Exclude them from the output
                if dt_ns_closed > 3_000_000: # 3 ms
                    continue
                
                segmentation_with_dt = gt_provider.get_segmentation_image_by_timestamp_ns(timestamp_ns, stream_id)
                
                if abs(segmentation_with_dt.dt_ns()) > 3_000_000: # 3 ms
                    continue

                # assert segmentation_with_dt.is_valid(), "segmentation not valid for input timestamp!"
                if not segmentation_with_dt.is_valid():
                    continue

                # RGB image
                image = image_with_dt.data().to_numpy_array()
                image = np.repeat(image[..., np.newaxis], 3, axis=2) if len(image.shape) < 3 else image
                
                rectified_image = calibration.distort_by_calibration(image, dst_calib, src_calib)

                # Segmentation
                segmentation_data = segmentation_with_dt.data().to_numpy_array()
                rectified_segmentation_data = calibration.distort_label_by_calibration(segmentation_data, dst_calib, src_calib)

                # Segmentation for visualization
                segmentation_viz = segmentation_with_dt.data().get_visualizable().to_numpy_array()
                rectified_segmentation_viz = calibration.distort_label_by_calibration(segmentation_viz, dst_calib, src_calib)

                # Save the output 
                if self.save_raw_images:
                    image_raw_output_subpath = f"images_raw/rgb_{timestamp_ns}.jpg"
                    if selected_device_number > 0: image_raw_output_subpath = f"dev{selected_device_number}_images_raw/rgb_{timestamp_ns}.jpg"
                    image_raw_output_path = output_folder / image_raw_output_subpath
                    os.makedirs(os.path.dirname(image_raw_output_path), exist_ok=True)
                    imageio.imwrite(image_raw_output_path, image)
                
                image_output_subpath = f"images/rgb_{timestamp_ns}.jpg"
                if selected_device_number > 0: image_output_subpath = f"dev{selected_device_number}_images/rgb_{timestamp_ns}.jpg"
                image_output_path = output_folder / image_output_subpath
                os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                imageio.imwrite(image_output_path, rectified_image)
                
                segmentation_output_subpath = f"masks/seg_{timestamp_ns}.pkl.gz"
                if selected_device_number > 0: segmentation_output_subpath = f"dev{selected_device_number}_masks/seg_{timestamp_ns}.pkl.gz"
                segmentation_output_path = output_folder / segmentation_output_subpath
                os.makedirs(os.path.dirname(segmentation_output_path), exist_ok=True)
                with gzip.open(segmentation_output_path, "wb") as f:
                    pickle.dump(rectified_segmentation_data, f)

                segmentation_viz_output_subpath = f"masks_viz/seg_{timestamp_ns}.jpg"
                if selected_device_number > 0: segmentation_viz_output_subpath = f"dev{selected_device_number}_masks_viz/seg_{timestamp_ns}.jpg"
                segmentation_viz_output_path = output_folder / segmentation_viz_output_subpath
                os.makedirs(os.path.dirname(segmentation_viz_output_path), exist_ok=True)
                imageio.imwrite(segmentation_viz_output_path, rectified_segmentation_viz)

                output_frame_metadata_device.append({
                    "fx": output_fx,
                    "fy": output_fy,
                    "cx": output_cx,
                    "cy": output_cy,
                    "w": self.output_rgb_width,
                    "h": self.output_rgb_height,
                    "image_path": image_output_subpath,
                    "transform_matrix": transform_world_cam.tolist(),
                    "timestamp": timestamp_ns,
                    "exposure_duration_s": 1.0,
                    "gain": 1.0,
                    "camera_name": "rgb",
                    "segmentation_path": segmentation_output_subpath,
                    "segmentation_viz_path": segmentation_viz_output_subpath,
                    "device": selected_device_number,
                    "vignette_subpath": vignette_save_subpath,
                    "mask_subpath": mask_save_subpath,
                })

            print(f"There are {len(output_frame_metadata_device)}/{len(img_timestamps_ns)} valid frames in the output.")
            output_frame_metadata.extend(output_frame_metadata_device)
            
        # Save the metadata
        metadata_save_path = output_folder / "transforms.json"
        with open(metadata_save_path, "w") as f:
            json.dump({
                "camera_model": "pinhole",
                "skeleton_instances_ids": skeleton_instances_ids,
                "static_instances_ids": static_instances_ids,
                "dynamic_instances_idx": dynamic_instances_idx,
                "frames": output_frame_metadata,
            }, f, indent=4)
            
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessAriaDigitalTwin).main()
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Convert Aria VRS to NeRFstudio "transforms.json" and single-file images.

import argparse
import json
import os
import imageio
import rerun as rr
from pathlib import Path
from glob import glob
from dataclasses import dataclass
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# @manual=fbsource//arvr/python/pyvrs2:pyvrs2
# import pyvrs2
# import pyvrs as pyvrs2

# @manual=fbsource//arvr/python/sophus:sophus
import projectaria_tools.core.sophus as sophus
from scipy.spatial.transform import Rotation as R
from PIL import Image

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import mps, data_provider, calibration
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
)
import tyro

@dataclass 
class PoseSE3: 
    transform_world_device: sophus.SE3
    device_linear_velocity_device: np.array 
    angular_velocity_device: np.array 
    quality_score: float 
    tracking_timestamp: int

def interpolate_closeloop_trajectory_pose(
        start_pose: mps.ClosedLoopTrajectoryPose, 
        end_pose: mps.ClosedLoopTrajectoryPose, 
        time_ns: int
    ):
    start_time = start_pose.tracking_timestamp.total_seconds() * 1e9
    end_time = end_pose.tracking_timestamp.total_seconds() * 1e9
    ratio = (time_ns - start_time) / (end_time - start_time)

    assert ratio >= 0.0 and ratio <= 1.0, f"interpolation ratio {ratio} is not within [0.0, 1.0]"
    interp_pose_SE3 = sophus.interpolate(start_pose.transform_world_device, end_pose.transform_world_device, ratio)
    interp_linear_velocity = start_pose.device_linear_velocity_device * (1-ratio) + end_pose.device_linear_velocity_device * ratio
    interp_angular_velocity = start_pose.angular_velocity_device * (1-ratio) + end_pose.angular_velocity_device * ratio
    interp_pose_score = start_pose.quality_score * (1-ratio) + end_pose.quality_score * ratio

    return PoseSE3(
        transform_world_device = interp_pose_SE3, 
        device_linear_velocity_device = interp_linear_velocity, 
        angular_velocity_device= interp_angular_velocity, 
        quality_score = interp_pose_score, 
        tracking_timestamp = int(time_ns)
    )


def interpolate_aria_pose(closed_loop_traj, capture_time_ns) -> PoseSE3: 
    start_idx = bisection_timestamp_search(closed_loop_traj, capture_time_ns)
    if start_idx is None: 
        return None 
    
    if closed_loop_traj[start_idx].tracking_timestamp.total_seconds() * 1e9 > capture_time_ns:
        start_idx -= 1

    start_pose = closed_loop_traj[start_idx]

    if start_idx + 1 >= len(closed_loop_traj): 
        return closed_loop_traj[start_pose]

    end_pose = closed_loop_traj[start_idx+1]

    interp_pose = interpolate_closeloop_trajectory_pose(start_pose, end_pose, time_ns=capture_time_ns)

    return interp_pose


@dataclass
class AriaImageFrame:
    camera: calibration.CameraCalibration       # Camera calibration 
    file_path: str                              # file path 
    image_size: np.array                        # (H, W)
    t_world_camera: sophus.SE3                  # The RGB camera to world transformation in SE3
    timestamp: float                            # Timestamp 
    device_linear_velocity: np.array            # The linear velocity of RGB camera (in device coordinate)
    device_angular_velocity: np.array           # The angular velocity of RGB camera (in device coordinate)
    exposure_duration_s: float                  # The exposure duration in second 
    gain: float                                 # The (analog) gain from device 
    time_diff_ns: int 

CAMERA_NAME_MAPPING = {
    "camera-rgb": "rgb",
    "camera-slam-left": "slaml",
    "camera-slam-right": "slamr",
}

def bisection_timestamp_search(timed_data, query_timestamp_ns: int) -> int:
    """
    Binary search helper function, assuming that timed_data is sorted by the field names 'tracking_timestamp'
    Returns index of the element closest to the query timestamp else returns None if not found (out of time range)
    """
    # Deal with border case
    if timed_data and len(timed_data) > 1:
        first_timestamp = timed_data[0].tracking_timestamp.total_seconds() * 1e9
        last_timestamp = timed_data[-1].tracking_timestamp.total_seconds() * 1e9
        if query_timestamp_ns <= first_timestamp:
            return None
        elif query_timestamp_ns >= last_timestamp:
            return None
    
    # If this is safe we perform the Bisection search
    timestamps_ns = [_.tracking_timestamp.total_seconds() * 1e9 for _ in timed_data]
    nearest_idx = np.searchsorted(timestamps_ns, query_timestamp_ns)  # a[i-1] < v <= a[i]
    
    # decide between a[i-1] and a[i]
    if nearest_idx > 0:
        start_minus_1_timestamp = timestamps_ns[nearest_idx - 1]
        start_timestamp = timestamps_ns[nearest_idx]
        if abs(start_minus_1_timestamp - query_timestamp_ns) < abs(start_timestamp - query_timestamp_ns):
            nearest_idx = nearest_idx - 1
            
    return nearest_idx

def get_nearest_pose(
    mps_trajectory: List[mps.ClosedLoopTrajectoryPose], query_timestamp_ns: int
) -> mps.ClosedLoopTrajectoryPose:
    """
    Helper function to get nearest pose for a timestamp (ns)
    Return the closest or equal timestamp pose information that can be found, returns None if not found (out of time range)
    """
    bisection_index = bisection_timestamp_search(mps_trajectory, query_timestamp_ns)
    if bisection_index is None:
        return None
    return mps_trajectory[bisection_index]

def to_aria_image_frame(
    provider, 
    online_camera_calibs: List[mps.OnlineCalibration],
    closed_loop_traj: mps.ClosedLoopTrajectoryPose,
    img_out_dir: str,
    camera_label: str = "camera-rgb",
    visualize: bool = True,
):
    assert camera_label in ["camera-rgb", "camera-slam-left", "camera-slam-right"]
    camera_label_short = CAMERA_NAME_MAPPING[camera_label]
    # todo: support raw mode. To determine if it is for raw files. 
    # config = provider.get_image_configuration(rgb_stream_id)
    # if config.pixel_format == 17: # raw mode
    #     read_raw_image = True 
    # else: 
    #     read_raw_image = False
    
    # #todo: Pickle does not support to multi-threading process this function. 
    def process_raw_data(frame_i: int, camera_label: str):
        stream_id = provider.get_stream_id_from_label(camera_label)

        sensor_data  = provider.get_sensor_data_by_index(stream_id, frame_i)
        image_data, image_record = sensor_data.image_data_and_record()

        if image_data.get_height() == 2880: 
            capture_time_offset = 8 * 1e6
        elif image_data.get_height() == 1408: 
            capture_time_offset = 2.5 * 1e6
        else: 
            raise RuntimeError(f"Unknow image data size! {image_data.get_height()}")

        # rgb camera timestamp offset
        # todo: make it more general also supporting 1408 (will be 2.5 ms)
        # capture_time_ns = image_record.capture_timestamp_ns
        capture_time_ns = int(image_record.capture_timestamp_ns + capture_time_offset)
        exposure_duration_s = image_record.exposure_duration
        gain = image_record.gain
        frame_nubmer = image_record.frame_number

        # Interpolate from two nearest poses based on the timestamp
        pose_info = interpolate_aria_pose(closed_loop_traj, capture_time_ns)
        if pose_info is None: return None
        camera_pose_time_diff_ns = 0
        
        # # Get the nearest pose 
        # pose_info = get_nearest_pose(closed_loop_traj, capture_time_ns)
        # if pose_info is None: return None
        # pose_time_ns = pose_info.tracking_timestamp.total_seconds() * 1e9
        # camera_pose_time_diff_ns = abs(pose_time_ns - capture_time_ns)
    
        if pose_info.quality_score < 0.9: 
            print(f"pose quality score below 0.9: {pose_info.quality_score}!")

        t_world_device = pose_info.transform_world_device

        # Get the online calibration
        nearest_calib_idx = bisection_timestamp_search(online_camera_calibs, capture_time_ns)
        if nearest_calib_idx is None: return None
        camera_calibration = online_camera_calibs[nearest_calib_idx]
        calib_time_ns = camera_calibration.tracking_timestamp.total_seconds() * 1e9
        camera_calib_time_diff_ns = abs(calib_time_ns - capture_time_ns)

        # find the one that is RGB camera
        camera_calib = None 
        for calib in camera_calibration.camera_calibs: 
            if calib.get_label() == camera_label: 
                camera_calib = calib 
                break 
        assert camera_calib is not None, "Did not find camera-rgb calibration in online calibrations!"

        # # Use the factory calibration
        # device_calib = provider.get_device_calibration()
        # camera_calib = device_calib.get_camera_calib(camera_label)

        # Gaussian Splatting (COLMAP) use the same coordinate system as Aria
        # Therefore there is no coordinate system conversion needed.
        t_world_camera = t_world_device @ camera_calib.get_transform_device_camera()

        image = image_data.to_numpy_array()
        os.makedirs(f"{img_out_dir}", exist_ok=True)
        # always store the images using png file. 
        img_file_path = f"{img_out_dir}/{camera_label_short}_{capture_time_ns}.png"
        Image.fromarray(image).save(img_file_path)

        return AriaImageFrame(
                camera=camera_calib,
                file_path=img_file_path,
                image_size=image.shape[:2],
                t_world_camera=t_world_camera,
                device_linear_velocity=pose_info.device_linear_velocity_device, 
                device_angular_velocity=pose_info.angular_velocity_device,
                timestamp=capture_time_ns,
                exposure_duration_s=exposure_duration_s,
                gain=gain,
                time_diff_ns=camera_pose_time_diff_ns,
            )

    frames = []
    stream_id = provider.get_stream_id_from_label(camera_label)

    num_process = 1
    total_frames = provider.get_num_data(stream_id)
    for frame_i in tqdm(range(0, total_frames, num_process)):
        # img_frame = process_raw_data(frame_i, camera_label=camera_label)
        num_process_to_launch = min(total_frames - frame_i, num_process)
        with ThreadPoolExecutor(max_workers=num_process) as e:
            futures = [e.submit(process_raw_data, frame_i+i, camera_label) for i in range(num_process_to_launch)]
            results = [future.result() for future in futures if future.result() is not None]
            sorted_results = sorted(results, key=lambda x: x.timestamp)

            # if img_frame is None: continue 
            # aria_image_frames.append(img_frame)
            for img_frame in sorted_results:
                frames.append(to_nerfstudio_frame(img_frame, visualize=visualize))

    ns_frames = {
        "camera_model": "FISHEYE624",
        "frames": frames,
    }
    print(f"{camera_label}: a total of {len(frames)} number of frames.")

    # with multiprocessing.Pool(24) as pool:
    #     pool_args = [(frame_i) for frame_i in range(0, provider.get_num_data(rgb_stream_id))]
    #     aria_image_frames = pool.starmap(process_raw_data, pool_args)
    # aria_image_frames = [x for x in aria_image_frames if x is not None]

    return ns_frames


def to_nerfstudio_frame(frame: AriaImageFrame, visualize: bool=True, scale: float=1000) -> Dict:

    fx, fy = frame.camera.get_focal_lengths()
    cx, cy = frame.camera.get_principal_point()
    h, w  = frame.image_size # the calibration image size might be incorrect due to API issue.

    camera_name = frame.camera.get_label()

    if visualize:
        rr.log(
            f"world/device",
            rr.Transform3D(
                translation=frame.t_world_camera.translation() * scale,
                mat3x3=frame.t_world_camera.rotation().to_matrix(),
            ),
        )
        # rotate image just for visualization 
        image = np.array(Image.open(frame.file_path).rotate(270))
        rr.log(
            f"world/device/{camera_name}/image",
            rr.Image(image).compress(jpeg_quality=75),
        )
        rr.log(
            f"world/device/{camera_name}",
            rr.Pinhole(resolution=[w, h], focal_length=[fx, fy]),
        )
        rr.log(
            f"world/device/{camera_name}_exposure_ms", 
            rr.Scalar(frame.exposure_duration_s * 1000)   
        )
        rr.log(
            f"world/device/{camera_name}_gain", 
            rr.Scalar(frame.gain)
        )

    cam_name_short = CAMERA_NAME_MAPPING[camera_name]
    
    return {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "distortion_params": frame.camera.projection_params()[3:15].tolist(),
        "w": int(w),
        "h": int(h),
        "file_path": frame.file_path,
        "camera_modality": 'rgb',
        "transform_matrix": frame.t_world_camera.to_matrix().tolist(),
        "timestamp": frame.timestamp,
        "time_diff_ns": frame.time_diff_ns,
        "exposure_duration_s": frame.exposure_duration_s,
        "gain": frame.gain,
        "camera_name": cam_name_short,
        "device_linear_velocity": frame.device_linear_velocity.tolist(), 
        "device_angular_velocity": frame.device_angular_velocity.tolist(),
    }


def visualize_frames(folder: Path, scale=100.0) -> None:

    rr.init(f"Visualize all frames rectified.", spawn=True)

    # VIsualize the transformed output path 
    transform_output_path = folder / "transforms.json"
    semidense_points3d_path = folder / "semidense_points.csv.gz"

    semidense_points_data = mps.read_global_point_cloud(str(semidense_points3d_path))

    filtered_point_positions = []
    for point in semidense_points_data:
        if point.inverse_distance_std < 0.001 and point.distance_std < 0.15:
            filtered_point_positions.append(point.position_world * scale) 
    rr.log(
        f"world/points_3D",
        rr.Points3D(filtered_point_positions, colors=[200, 200, 200], radii=0.01),
        timeless=True,
    )

    with open(transform_output_path, "r") as f: 
        ns_frames = json.load(f)
        frames = ns_frames["frames"]

    for _, frame in enumerate(frames):

        # rr.set_time_sequence("frame_idx", frame_idx)
        rr.set_time_seconds("sensor_time", frame.timestamp / 1e9)

        transform_matrix = np.array(frame["transform_matrix"])
        rr.log(
            f"world/device",
            rr.Transform3D(
                translation=transform_matrix[:3, 3] * scale,
                mat3x3=transform_matrix[:3, :3],
            ),
        )

        image = np.array(Image.open(folder / frame["image_path"]))
        rr.log(
            f"world/device/rgb/image",
            rr.Image(image).compress(jpeg_quality=75),
        )

        rr.log(
            f"world/device/rgb",
            rr.Pinhole(
                resolution=[frame["w"], frame["h"]],
                focal_length=float(frame["fx"]),
            ),
        )

        if frame["mask_path"] != "":
            mask = np.array(Image.open(folder / frame["mask_path"]))
            rr.log(f"world/device/rgb/mask", rr.SegmentationImage(mask))

        rr.log(
            f"world/device/exposure_ms", 
            rr.Scalar(frame["exposure_duration_s"] * 1000)   
        )
        rr.log(
            f"world/device/gain", 
            rr.Scalar(frame["gain"])
        )


def run_single_sequence(
    recording_folder: str, 
    vrs_file: Path,
    trajectory_file: Path, 
    online_calibration_file: Path,
    semi_dense_points_file: Path,
    semi_dense_observation_file: Path,
    output_path: Path,
    process_rgb: bool,
    process_slam: bool,
    options, # all other options in configs
):
    if vrs_file == "":
        input_vrs = glob(str(recording_folder / "*.vrs"))
        assert len(input_vrs) == 1, "the target folder should only have 1 vrs file."
        input_vrs = Path(input_vrs[0])
        print(f"Find VRS file: {input_vrs}")
    else: 
        input_vrs = recording_folder / vrs_file

    assert input_vrs.exists(), f"cannot find input vrs file {input_vrs}"

    if options.visualize:
        rr.init(f"Extract VRS file from {vrs_file}", spawn=True)

    print("Getting poses from closed loop trajectory CSV...")

    assert trajectory_file.exists(), f"cannot find trajectory file {trajectory_file}"
    closed_loop_traj = mps.read_closed_loop_trajectory(str(trajectory_file))
 
    print("Get semi-dense point cloud")
    semidense_points_data = mps.read_global_point_cloud(str(semi_dense_points_file))
    inverse_distance_std_threshold = 0.001
    distance_std_threshold = 0.15
    filtered_semidense_points = filter_points_from_confidence(
        semidense_points_data, inverse_distance_std_threshold, distance_std_threshold
    )
    scale = 1000.0 # we will use this hard-coded parameter for all.
    point_positions = [it.position_world * scale for it in filtered_semidense_points]

    if options.visualize:
        rr.log(
            f"world/points_3D",
            rr.Points3D(point_positions, colors=[200, 200, 200], radii=0.01),
            timeless=True,
        )

    assert online_calibration_file.exists(), f"cannot find online calibration file {online_calibration_file}"
    camera_calibs = mps.read_online_calibration(str(online_calibration_file))

    vrs_provider = data_provider.create_vrs_data_provider(str(input_vrs))

    # create an AriaImageFrame for each image in the VRS.
    camera_process_list = []
    if process_rgb: 
        camera_process_list.append("camera-rgb")
    if process_slam: 
        camera_process_list.append("camera-slam-left")
        camera_process_list.append("camera-slam-right")

    semidense_observations = mps.read_point_observations(str(semi_dense_observation_file))

    ########################################
    # Process Transform json file
    ########################################
    frames_all = {
        "camera_model": "FISHEYE624",
        "frames": [],
    }
    for camera_label in camera_process_list:
        frames = to_aria_image_frame(
            provider=vrs_provider, 
            online_camera_calibs=camera_calibs, 
            closed_loop_traj=closed_loop_traj, 
            img_out_dir=str(output_path / f"images_raw"), 
            camera_label=camera_label,
            visualize=options.visualize,
        )
        assert frames['camera_model'] == "FISHEYE624", "Only support FISHEYE624 camera model."
        frames_all['frames'] += frames['frames']
        
    json_path = output_path / f"transforms_raw.json"
    print(f"Write camera information for {camera_label} to {json_path}")
    with open(json_path, "w", encoding="UTF-8") as file: 
        json.dump(frames_all, file, indent=4)

    # set up a symbolic link for the semi-dense point cloud
    semidense_points_path_in_rectified = output_path / "semidense_points.csv.gz"
    if not semidense_points_path_in_rectified.exists():
        os.symlink(semi_dense_points_file, semidense_points_path_in_rectified)


def bilinear_sample(image, u, v): 
    u_fl = int(np.floor(u))
    v_fl = int(np.floor(v))
    w_u = 1 - (u - u_fl)
    w_v = 1 - (v - v_fl)
    rgb = w_u * w_v * image[v_fl, u_fl] + \
        w_u * (1-w_v) * image[v_fl+1, u_fl] + \
        (1-w_u) * w_v * image[v_fl, u_fl+1] + \
        (1-w_u)*(1-w_v) * image[v_fl+1, u_fl+1]
    return rgb


@dataclass
class ProcessProjectAria:
    """Processes Project Aria data i.e. a VRS of the raw recording streams and the MPS attachments
    that provide poses, calibration, and 3d points. More information on MPS data can be found at:
      https://facebookresearch.github.io/projectaria_tools/docs/ARK/mps.
    """

    vrs_folder: Path
    """Path to the folder where vrs file resides."""
    mps_data_dir: Path
    """Path to Project Aria Machine Perception Services (MPS) attachments."""
    output_dir: Path
    """Path to the output directory."""
    
    vrs_file: str = ""
    """Path to the VRS file."""
    
    trajectory_file: Optional[Path] = None
    """Name of the trajectory file."""
    online_calib_file: Optional[Path] = None
    """Name of the online calibration file."""
    semi_dense_points_file: Optional[Path] = None
    """Name of the semi-dense point cloud file."""
    semi_dense_observation_file: Optional[Path] = None
    """Name of the semi-dense observation file."""
    
    process_rgb: bool = True
    """Whether to process RGB camera."""
    process_slam: bool = False
    """Whether to process SLAM camera."""
    
    visualize: bool = False
    """Whether to visualize the output."""
    
    def main(self) -> None:
        # Create output directory if it doesn't exist.
        self.output_dir = self.output_dir.absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.trajectory_file is None:
            trajectory_file = self.mps_data_dir / "closed_loop_trajectory.csv"
        else: 
            trajectory_file = self.trajectory_file
        print(f"Will load trajectory file: {trajectory_file}")

        if self.online_calib_file is None:
            online_calib_file = self.mps_data_dir / "online_calibration.jsonl"
        else: 
            online_calib_file = self.online_calib_file
        print(f"Will load calibration file: {online_calib_file}")
        
        if self.semi_dense_points_file is None:
            semidense_points_path = self.mps_data_dir / "semidense_points.csv.gz"
        else:
            semidense_points_path = self.semi_dense_points_file
        print(f"Will load semi-dense point cloud file: {semidense_points_path}")

        if self.semi_dense_observation_file is None:
            semidense_observation_path = self.mps_data_dir / "semidense_observations.csv.gz"
        else: 
            semidense_observation_path = Path(self.semi_dense_observation_file)
        
        run_single_sequence(
            recording_folder = self.vrs_folder,
            vrs_file = self.vrs_file,
            trajectory_file = trajectory_file, 
            online_calibration_file = online_calib_file,
            semi_dense_points_file = semidense_points_path,
            semi_dense_observation_file = semidense_observation_path,
            output_path = self.output_dir, 
            process_rgb = self.process_rgb,
            process_slam = self.process_slam,
            options=self,
        )
    
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessProjectAria).main()


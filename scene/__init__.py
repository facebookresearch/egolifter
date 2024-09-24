# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco


from argparse import Namespace
import copy
import gzip
import os
import pickle
import json
import uuid
import glob
import warnings

from PIL import Image
from natsort import natsorted
import numpy as np
import torch

from scipy.spatial.transform import Rotation

from torch.utils.data import Dataset

from utils.general_utils import PILtoTorch
from scene.dataset_readers import SceneType
from scene.dataset_readers import get_scene_info, storePly, aggregate_scene_infos
from scene.gaussian_model import GaussianModel
from scene.gaussian_union import GaussianUnion
from utils.camera_utils import getResolution, cam_pose_to_GS_Rt, GS_Rt_to_cam_pose
from scene.cameras import camera_to_JSON, Camera

from utils.constants import BAD_ARIA_PILOT_SCENES
from utils.contrastive import ContrastManager


class CameraDataset(Dataset):
    '''
    This Dataset is to load images, mask and other Tensors that may cause OOM if loaded all at once
    The Camera object cannot be handled by this class but you can get it by scene.get_camera()

    An example usage:
    ```
    for batch in train_loader:
        gt_image = batch['image'].to("cuda")[0]
        scene.get_camera(batch['idx'].item(), subset="train")
    '''
    def __init__(
        self, 
        cameras: list[Camera], 
        full_res, 
        scene_info,
        valid_mask_by_name,
        vignette_by_name,
        name="train", 
        scale=1.0,
        contrast_manager=None,
    ):
        self.cameras = cameras
        self.full_res = full_res
        self.scene_info = scene_info
        self.valid_mask_by_name = copy.deepcopy(valid_mask_by_name)
        self.vignette_by_name = copy.deepcopy(vignette_by_name)
        self.name = name
        self.scale = scale
        self.contrast_manager = contrast_manager
        
        # Convert the masks to CPU
        for n in self.valid_mask_by_name.keys():
            for scale in self.valid_mask_by_name[n].keys():
                self.valid_mask_by_name[n][scale] = self.valid_mask_by_name[n][scale].cpu()
        
        for n in self.vignette_by_name.keys():
            for scale in self.vignette_by_name[n].keys():
                self.vignette_by_name[n][scale] = self.vignette_by_name[n][scale].cpu()
        
    def get_camera(self, idx: int) -> Camera:
        return self.cameras[idx]

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        cam = self.cameras[idx]
        
        image = Image.open(cam.image_path)
        resolution = getResolution(self.full_res, image, self.scale)
        resized_image_rgb = PILtoTorch(image, resolution)

        # If the image has an alpha channel, extract it as the mask
        gt_image = resized_image_rgb[:3, ...]
        
        batch = {
            # meta info
            "idx": idx, 
            "subset": self.name,
            "scene_name": cam.scene_name,
            "image_name": cam.image_name,
            "image_id": cam.uid,
            # torch tensors
            "image": gt_image,
            "fid": cam.fid,
        }
        
        if resized_image_rgb.shape[0] == 4:
            alpha_mask = resized_image_rgb[3:4, ...]
            batch["alpha_mask"] = alpha_mask
        
        # Load the valid mask. If the dict is empty, return a mask full of ones
        valid_mask = torch.ones(1, resolution[1], resolution[0])
        if self.valid_mask_by_name: # If the dict is not empty
            valid_mask = self.valid_mask_by_name[cam.valid_mask_subpath][self.scale] # (1, H, W), in [0.0, 1.0

        # valid_mask = self.valid_mask_by_name[cam.valid_mask_subpath][self.scale] # (1, H, W), in [0.0, 1.0]
        # vignette = self.vignette_by_name[cam.vignette_subpath][self.scale] # (3, H, W)
        
        batch['valid_mask'] = valid_mask
        
        # Load the GT 2D segmentation image
        if cam.seg_path is not None:
            if cam.seg_path.endswith(".pkl.gz"):
                with gzip.open(cam.seg_path, "rb") as f:
                    seg_mask = pickle.load(f) # (1408, 1408)
                dynamic_mask = np.isin(seg_mask, self.scene_info.seg_dynamic_ids)
                dynamic_mask = torch.from_numpy(dynamic_mask).float().unsqueeze(0) # (1, 1408, 1408), in {0, 1}
            else:
                raise NotImplementedError(f"Segmentation mask {cam.seg_path} is not supported yet.")
            
            # Compute the masks for dynamic and static objects
            batch["dynamic_mask"] = dynamic_mask * valid_mask
            batch['static_mask'] = (1.0 - dynamic_mask) * valid_mask
            
            batch['seg_mask'] = torch.from_numpy(seg_mask.astype(np.int64)).unsqueeze(0) # (1, 1408, 1408)

        if self.scene_info.query_2dseg is not None:
            batch['query_2dseg'] = self.scene_info.query_2dseg[cam.image_name]
            batch['seg_dynamic_ids'] = self.scene_info.seg_dynamic_ids
            batch['seg_static_ids'] = self.scene_info.seg_static_ids
            
        # Load the 2D masks from segmentation network (SAM)
        if self.contrast_manager is not None and self.contrast_manager.in_use:
            mask_image = self.contrast_manager.get_mask(cam)
            batch['mask_image'] = mask_image
        
        return batch

class Scene:

    gaussians : GaussianModel

    def __init__(
            self, 
            cfg, 
            resolution_scales=[1.0], 
            scene_info=None,
            simple: bool = False, # If set, this class will be initialized in a simple way
        ):
        
        self.cfg = cfg
        self.data_root = cfg.scene.data_root
        self.scene_name = cfg.scene.scene_name
        self.model_path = cfg.scene.model_path
        self.loaded_iter = None
        
        if cfg.scene.aggregate:
            # Aggregate data from multiple scans
            source_paths = [d for d in glob.glob(cfg.scene.source_path+"*/") if os.path.isdir(d)]
        else:
            source_paths = [cfg.scene.source_path]
            
        assert len(source_paths) > 0, f"No source path found in {cfg.scene.source_path}"
        if len(source_paths) == 1:
            self.scene_info = get_scene_info(cfg.scene) if scene_info is None else scene_info
            self.scene_type = self.scene_info.scene_type
            self.source_path = source_paths[0]
        else: 
            print(f"There are {len(source_paths)} paths provided. Will extract and merge them in one scene.")
            scenes_info = []
            original_scene_name = cfg.scene.scene_name
            original_source_path = cfg.scene.source_path
            self.scene_type = None
            for source_path in source_paths: 
                scene_name = source_path.split("/")[-2]
                if scene_name in BAD_ARIA_PILOT_SCENES:
                    print(f"Skipping {source_path}")
                    continue
                
                print(f"load and aggregated {source_path}")
                cfg.scene.scene_name = scene_name
                cfg.scene.source_path = source_path
                scene_info = get_scene_info(cfg.scene)
                scenes_info.append(scene_info)

                if self.scene_type is None:
                    self.scene_type = scene_info.scene_type
                else:
                    assert self.scene_type == scene_info.scene_type, f"scene type mismatch: {self.scene_type} vs {scene_info.scene_type}"
                
            cfg.scene.scene_name = original_scene_name
            cfg.scene.source_path = original_source_path

            # merge the scene infos into one 
            self.scene_info = aggregate_scene_infos(scenes_info)
            
            # TODO: handle multiple source paths
            self.source_path = None
            
        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        self.train_cameras = self.scene_info.train_cameras
        self.valid_cameras = self.scene_info.valid_cameras
        self.test_cameras = self.scene_info.test_cameras
        self.valid_novel_cameras = list(self.scene_info.valid_novel_cameras)

        self.trainvalid_cameras = self.train_cameras + self.valid_cameras
        self.trainvalid_cameras = natsorted(self.trainvalid_cameras, key=lambda x: x.image_name)

        self.all_cameras = self.trainvalid_cameras + self.test_cameras
        self.all_cameras = natsorted(self.all_cameras, key=lambda x: x.image_name)
        
        self.novel_cameras = self.valid_novel_cameras + self.test_cameras
        self.novel_cameras = natsorted(self.novel_cameras, key=lambda x: x.image_name)
        
        self.subset_to_cameras = {
            "train": self.train_cameras,
            "valid": self.valid_cameras,
            "valid_novel": self.valid_novel_cameras,
            "test": self.test_cameras,
            "trainvalid": self.trainvalid_cameras,
            "novel": self.novel_cameras,
            "all": self.all_cameras,
        }
        
        for subset, cameras in self.subset_to_cameras.items():
            print(f"Found {len(cameras)} images for {subset} subset.")
            
        self.extractMaskVignette(cfg.scene, self.scene_info, resolution_scales)

        if simple:
            warnings.warn("This scene object is initialized in a simple way. Training may fail.")
            self.contrast_manager = None
        else:
            # Initialize a help class for segmentation contrastive lifting
            self.contrast_manager = ContrastManager(
                cfg, 
                example_cam=self.train_cameras[0],
                valid_mask_by_name=self.valid_mask_by_name,
                scene_type=self.scene_type,
            )

            # Compute a normalization factor for exposure
            exposure_gain = []
            for cam in self.train_cameras:
                exposure_gain.append(cam.exposure * cam.gain)
            self.mean_exposure_gain = np.mean(exposure_gain)

            self.gamma = torch.tensor(1.0 / 2.2).to("cuda")
        
    def save_camera_json(self):
        # Save the training and test cameras
        camera_json_save_path = os.path.join(self.model_path, "cameras.json")
        if not os.path.exists(camera_json_save_path):
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(camera_json_save_path, 'w') as file:
                json.dump(json_cams, file)
        
    
    def get_camera(self, idx, subset):
        camera = copy.deepcopy(self.subset_to_cameras[subset][idx])
        return camera
        

    def get_data_loader(self, subset, scale=1.0, shuffle=False, num_workers=12, limit = None):
        cameras = self.subset_to_cameras[subset]
        
        # limit parameter may make get_camera() return different camera objects
        # So if using limit, do not use get_camera() of this scene object. Use that in the CameraDataset instead.
        if limit is not None:
            if limit < len(cameras):
                print(f"Subsampling {subset} set from {len(cameras)} to {limit} images.")
                subsample_indices = np.linspace(0, len(cameras)-1, limit).astype(int)
                cameras = [cameras[i] for i in subsample_indices]
            
        dataset = CameraDataset(
            cameras, 
            self.cfg.scene.resolution, 
            self.scene_info,
            self.valid_mask_by_name,
            self.vignette_by_name,
            name = subset,
            scale = scale,
            contrast_manager=self.contrast_manager,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=shuffle, 
            num_workers=num_workers,
        )
        return dataloader
        
    def get_camera_poses(self, subset = "train"):
        cameras = self.subset_to_cameras[subset]
            
        camera_poses = []
        for i in range(len(cameras)):
            camera_poses.append(GS_Rt_to_cam_pose(cameras[i].R, cameras[i].T))
        camera_poses = np.stack(camera_poses, axis=0)
        
        return camera_poses
        
    def get_scene_cam_time(self):
        '''
        Compute the various statistics of the scene, return a list of dict that captures the camera motion and coverage
        and is aready to be logged using wandb
        '''
        scene_cam_logs = []
        camera_poses = self.get_camera_poses("train")
        
        scene_cam_logs.append({
            "train_cam/range_x": camera_poses[:, 0, 3].max() - camera_poses[:, 0, 3].min(),
            "train_cam/range_y": camera_poses[:, 1, 3].max() - camera_poses[:, 1, 3].min(),
            "train_cam/range_z": camera_poses[:, 2, 3].max() - camera_poses[:, 2, 3].min(),
        })
        
        for i in range(len(camera_poses)):
            log = {}
            log["train_cam/abs_id"] = i
            log["train_cam/rel_id"] = float(i) / len(camera_poses)
            log["train_cam/pos_x"] = camera_poses[i, 0, 3]
            log["train_cam/pos_y"] = camera_poses[i, 1, 3]
            log["train_cam/pos_z"] = camera_poses[i, 2, 3]
            euler = Rotation.from_matrix(camera_poses[i, :3, :3]).as_euler('xyz', degrees=True)
            log["train_cam/rot_x"] = euler[0]
            log["train_cam/rot_y"] = euler[1]
            log["train_cam/rot_z"] = euler[2]
            
            if 0 < i < len(camera_poses) - 1:
                log["train_cam/vel_x"] = (camera_poses[i+1, 0, 3] - camera_poses[i-1, 0, 3]) / 2.0
                log["train_cam/vel_y"] = (camera_poses[i+1, 1, 3] - camera_poses[i-1, 1, 3]) / 2.0
                log["train_cam/vel_z"] = (camera_poses[i+1, 2, 3] - camera_poses[i-1, 2, 3]) / 2.0
                
            scene_cam_logs.append(log)
            
        return scene_cam_logs
        

    def extractMaskVignette(self, cfg_scene, scene_info, resolution_scales):
        '''
        Extracts the mask and vignette for the scene
        only useful on the Aria dataset
        '''
        self.valid_mask_by_name = None
        if scene_info.valid_mask_by_name is not None:
            self.valid_mask_by_name = {}
            for n, image in scene_info.valid_mask_by_name.items():
                self.valid_mask_by_name[n] = {}
                for resolution_scale in resolution_scales:
                    resolution = getResolution(cfg_scene.resolution, image, resolution_scale)
                    mask = PILtoTorch(image, resolution).to("cuda")
                    mask = (mask > 0.5).to(mask)
                    self.valid_mask_by_name[n][resolution_scale] = mask
        
        self.vignette_by_name = None
        if scene_info.vignette_by_name is not None:
            self.vignette_by_name = {}
            for n, image in scene_info.vignette_by_name.items():
                if image is None:
                    continue
                self.vignette_by_name[n] = {}
                for resolution_scale in resolution_scales:
                    resolution = getResolution(cfg_scene.resolution, image, resolution_scale)
                    self.vignette_by_name[n][resolution_scale] = PILtoTorch(image, resolution).to("cuda")
                    
            
    def postProcess(
            self, 
            render_image: torch.Tensor, 
            gt_image: torch.Tensor, 
            camera: Camera, 
            scale = 1.0
        ):
        '''
        Post process Gt and rendered images
        Only effective on the Aria dataset
        '''
        if (self.vignette_by_name) and (camera.vignette_subpath is not None):
            vignette = self.vignette_by_name[camera.vignette_subpath][scale] # (3, H, W)
            if self.scene_type == SceneType.ARIA and self.cfg.scene.use_hdr:
                render_image = render_image * camera.exposure * camera.gain * vignette
                render_image = render_image / self.mean_exposure_gain
                
                # This small offset is important. Otherwise the gradient will be NaN even though no value is less than 0. 
                render_image = (render_image + 1e-7).pow(self.gamma)
            else:
                render_image = render_image * vignette
        
        if self.valid_mask_by_name:
            valid_mask = self.valid_mask_by_name[camera.valid_mask_subpath][scale] # (1, H, W)
            gt_image = gt_image * valid_mask
            render_image = render_image * valid_mask

        return render_image, gt_image

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self) -> list[Camera]:
        return self.train_cameras

    def getTestCameras(self) -> list[Camera]:
        return self.test_cameras
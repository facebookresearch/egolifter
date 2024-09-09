from datetime import datetime
import os
import copy
import threading
from pathlib import Path
import glob
import time
import json
import argparse
from typing import Tuple, Literal, List
import imageio
from natsort import natsorted

import numpy as np
import viser
import viser.transforms as vtf
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import KDTree

from scene import Scene
from model import get_model
from utils.system_utils import get_latest_ckpt

from viewer import ClientThread
from viewer.client import get_sizes
from viewer.ui import populate_render_tab, TransformPanel, EditPanel
from utils.graphics_utils import focal2fov
from utils.system_utils import get_all_ckpt_subpaths
from utils.routines import load_from_model_path
from utils.gaussians import convert_gaussian_feat_pca, improve_saturation

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"

def find_nearest_se3(ref_pose: np.ndarray, poses: np.ndarray) -> [int, np.ndarray]:
    """
    Find the nearest 6D pose in poses to ref_pose
    
    Args:
        ref_pose: (4, 4) np.ndarray, the reference pose
        poses: (N, 4, 4) np.ndarray, the poses to search from
        
    Returns:
        (4, 4) np.ndarray, the nearest pose
    """
    # Compute the Frobenius norm (matrix norm) between ref_pose and each pose in poses
    distances = np.linalg.norm(poses - ref_pose, ord='fro', axis=(-2, -1))
    
    # Find the index of the pose with the smallest distance to ref_pose
    nearest_index = np.argmin(distances)
    
    # Return the nearest pose
    return nearest_index.item(), poses[nearest_index]

def scan_avail_iters(model_path):
    # Scan and get all the checkpoints
    all_paths = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*"))
    all_paths = natsorted(all_paths)
    iter_names = [os.path.basename(p).replace("iteration_", "") for p in all_paths]
    return iter_names

class Viewer:
    def __init__(
            self,
            model_root: str,
            host: str = "0.0.0.0",
            port: int = 8080,
            image_format: Literal["jpeg", "png"] = "jpeg",
            reorient: Literal["auto", "enable", "disable"] = "auto",
            enable_transform: bool = False,
            show_cameras: bool = False,
            data_root: str = None,
            feat_pca: bool = False,
    ):
        self.device = torch.device("cuda")

        self.model_root = model_root
        self.data_root = data_root
        self.host = host
        self.port = port
        self.image_format = image_format
        self.sh_degree = 3
        self.enable_transform = enable_transform
        self.show_cameras = show_cameras
        self.reorient = reorient
        self.add_feat_pca = feat_pca
        
        self.lock_aspect_one = True
        
        if "adt" in model_root:
            self.aria_up_direction = np.asarray([0., 1., 0.])
        else:
            self.aria_up_direction = np.asarray([0., 0., 1.])

        self.up_direction = self.aria_up_direction
        
        # Recursively search for all the subfolders in model_root
        model_paths = glob.glob(os.path.join(model_root, "**", "point_cloud"), recursive=True)
        model_paths = [os.path.dirname(p) for p in model_paths]
        model_paths = sorted(model_paths)

        # Filter out the invalid model path by validating the existence of point_cloud folder
        self.model_paths = []
        self.model_names = []
        for p in model_paths:
            if os.path.isdir(p) is False:
                continue
            point_cloud_folder = os.path.join(p, "point_cloud")
            if os.path.exists(point_cloud_folder):
                self.model_paths.append(p)
                self.model_names.append(p[len(model_root):])
        print("Found {} models".format(len(self.model_paths)))
                
        model_idx = 0
        while True:
            # self.iter_names = scan_avail_iters(self.model_paths[model_idx])
            self.ckpt_subpaths = get_all_ckpt_subpaths(self.model_paths[model_idx])
            if len(self.ckpt_subpaths) > 0:
                break
            model_idx += 1
            if model_idx >= len(self.model_paths):
                raise RuntimeError("No valid model found")

        self.server = None
        self.clients = {}
        
        self.init_from_model_path(self.model_paths[0], self.ckpt_subpaths[-1])
        
        
    def init_from_model_path(self, model_path: str, ckpt_subpath: str = None):
        self.simplified_model = True
        self.show_edit_panel = True
        self.show_render_panel = True
        
        model, scene, cfg = load_from_model_path(
            model_path,
            data_root=self.data_root,
            ckpt_subpath=ckpt_subpath,
        )
        
        # Update the config according to the loaded model
        self.sh_degree = cfg.model.sh_degree
        if cfg.model.white_background:
            self.background_color = (1., 1., 1.)
        else:
            self.background_color = (0., 0., 0.)
        
        # reorient the scene
        cameras_json_path = os.path.join(model_path, "cameras.json")
        self.camera_transform = self._reorient(cameras_json_path, mode=self.reorient, dataset_type=None)
        # load camera poses
        self.camera_poses = self.load_camera_poses(cameras_json_path)
        self.camera_poses_viewer = self.get_camera_poses_viewer()

        self.available_appearance_options = None

        self.model = model
        
        # Get the PCA visualized features if needed
        self.model_feat_pca = None
        if self.add_feat_pca:
            with torch.no_grad():
                self.model_feat_pca = copy.deepcopy(self.model)
                convert_gaussian_feat_pca(self.model_feat_pca.gaussians, alpha=1.0)
                improve_saturation(self.model_feat_pca.gaussians, saturation_factor=5.0)
        
        self.prepare_query()
        
        # Register the model to clients
        self.using_pca = False # Flag about what model is currently being used
        for i in self.clients:
            self.clients[i].model = self.model

        print("Finished loading ckpt from", model_path, ckpt_subpath)
        
    
    @torch.no_grad()
    def prepare_query(self):
        gaussians = self.model.gaussians
        self.kdtree_gaussians = KDTree(gaussians.get_xyz.detach().cpu().numpy())
        self.gaussian_features = gaussians.get_features_extra.detach().cpu() # (N, D)
        self.original_opacity = gaussians._opacity.detach().clone()


    def _reorient(self, cameras_json_path: str, mode: str, dataset_type: str = None):
        transform = torch.eye(4, dtype=torch.float)

        if mode == "disable":
            return transform

        # detect whether cameras.json exists
        is_cameras_json_exists = os.path.exists(cameras_json_path)

        if is_cameras_json_exists is False:
            if mode == "enable":
                raise RuntimeError("{} not exists".format(cameras_json_path))
            else:
                return transform

        # skip reorient if dataset type is blender
        if dataset_type in ["blender", "nsvf"] and mode == "auto":
            print("skip reorient for {} dataset".format(dataset_type))
            return transform

        print("load {}".format(cameras_json_path))
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up = torch.zeros(3)
        for i in cameras:
            up += torch.tensor(i["rotation"])[:3, 1]
        up = -up / torch.linalg.norm(up)

        print("up vector = {}".format(up))
        self.up_direction = up

        return transform

        
    def load_camera_poses(self, cameras_json_path: str):
        if os.path.exists(cameras_json_path) is False:
            return []
        with open(cameras_json_path, "r") as f:
            return json.load(f)
        

    def add_cameras_to_scene(self, viser_server):
        if len(self.camera_poses) == 0:
            return

        self.camera_handles = []

        camera_pose_transform = np.linalg.inv(self.camera_transform.cpu().numpy())
        for camera in self.camera_poses:
            name = camera["img_name"]
            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(camera["rotation"])
            c2w[:3, 3] = np.asarray(camera["position"])
            c2w[:3, 1:3] *= -1
            c2w = np.matmul(camera_pose_transform, c2w)

            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)

            cx = camera["width"] // 2
            cy = camera["height"] // 2
            fx = camera["fx"]

            camera_handle = viser_server.add_camera_frustum(
                name="cameras/{}".format(name),
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.1,
                aspect=float(cx / cy),
                wxyz=R.wxyz,
                position=c2w[:3, 3],
                color=(205, 25, 0),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)

        self.camera_visible = True

        def toggle_camera_visibility(_):
            with viser_server.atomic():
                self.camera_visible = not self.camera_visible
                for i in self.camera_handles:
                    i.visible = self.camera_visible

        # def update_camera_scale(_):
        #     with viser_server.atomic():
        #         for i in self.camera_handles:
        #             i.scale = self.camera_scale_slider.value

        with viser_server.add_gui_folder("Cameras"):
            self.toggle_camera_button = viser_server.add_gui_button("Toggle Camera Visibility")
            # self.camera_scale_slider = viser_server.add_gui_slider(
            #     "Camera Scale",
            #     min=0.,
            #     max=1.,
            #     step=0.01,
            #     initial_value=0.1,
            # )
        self.toggle_camera_button.on_click(toggle_camera_visibility)
        # self.camera_scale_slider.on_update(update_camera_scale)
        
    
    def get_current_model_path(self):
        return self.model_root + self.model_paths_dropdown.value

    
    def get_camera_poses_viewer(self) -> np.ndarray:
        '''
        Get the matrix form of self.camera_poses
        '''
        pose_matrix = []
        for cam_pose in self.camera_poses:
            # Convert the camera pose from the dataloader format to web viewer format
            R = cam_pose['rotation']
            T = cam_pose['position']
            w2c = np.eye(4)
            w2c[:3, :3] = np.asarray(R)
            w2c[:3, 3] = np.asarray(T)

            # c2w = np.linalg.inv(w2c)
            c2w = w2c # TODO: figure out why this is correct
            
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            
            c2w = torch.tensor(c2w, dtype=torch.float)
            c2w = torch.linalg.inv(self.camera_transform) @ c2w
            c2w = c2w.numpy()
            c2w[:3, :3] = c2w[:3, :3] @ vtf.SO3.from_x_radians(-np.pi).as_matrix()

            # pose = vtf.SE3.from_matrix(c2w)
            pose_matrix.append(c2w)
            
        return np.asarray(pose_matrix)
    
    
    def add_capture_save_folder(self, server: viser.ViserServer):
        if hasattr(self, "capture_save_folder") and self.capture_save_folder is not None:
            self.capture_save_folder.remove()
            
        with server.add_gui_folder("Capture") as gui_folder:
            self.capture_save_folder = gui_folder
            save_name_textbox = server.add_gui_text("Save Name", initial_value="capture")
            capture_button = server.add_gui_button("Capture", icon=viser.Icon.CAMERA)
            
            @capture_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                camera = event.client.camera
                save_name = save_name_textbox.value

                max_res = self.max_res_when_static.value
                aspect_ratio = camera.aspect
                _, _, image_width, image_height = get_sizes(
                    max_res, 
                    aspect_ratio,
                    self.lock_aspect_one,
                )
                
                render = camera.get_render(
                    image_height, 
                    image_width,
                    "jpeg"
                )
                time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("./captures", exist_ok=True)
                save_path = os.path.join("./captures", f"{save_name}_{time_suffix}.jpg")
                imageio.imwrite(save_path, render)


    def add_frame_slider_folder(self, server: viser.ViserServer):
        if hasattr(self, "frame_slider_folder") and self.frame_slider_folder is not None:
            self.frame_slider_folder.remove()
        
        with server.add_gui_folder("Training Cameras") as gui_folder:
            self.frame_slider_folder = gui_folder
            # Add a slider to traverse through the training cameras
            play_button = server.add_gui_button("Play", icon=viser.Icon.PLAYER_PLAY)
            pause_button = server.add_gui_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
            snap_button = server.add_gui_button("Nearest camera", icon=viser.Icon.CAMERA, visible=True)
            prev_button = server.add_gui_button("Previous", icon=viser.Icon.ARROW_AUTOFIT_LEFT)
            next_button = server.add_gui_button("Next", icon=viser.Icon.ARROW_AUTOFIT_RIGHT)
            
            frame_step_slider = server.add_gui_slider(
                "Frame Step",
                min=1, 
                max=20,
                step=1,
                initial_value=5,
            )
            
            frame_slider = server.add_gui_slider(
                "Frame",
                min=0,
                max=len(self.camera_poses) - 1,
                step=1,
                initial_value=0,
            )
            self.frame_slider = frame_slider
            
            @frame_slider.on_update
            def _(_) -> None:
                cam_pose = self.camera_poses[frame_slider.value]
                pose = vtf.SE3.from_matrix(self.camera_poses_viewer[frame_slider.value])
                fov = focal2fov(cam_pose['fy'], cam_pose['height'])
                
                for client in server.get_clients().values():
                    with client.atomic():
                        client.camera.wxyz = pose.rotation().wxyz
                        client.camera.position = pose.translation()
                        client.camera.fov = fov
                        
            @snap_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                current_pose = np.eye(4)
                current_pose[:3, :3] = vtf.SO3(event.client.camera.wxyz).as_matrix()
                current_pose[:3, 3] = event.client.camera.position
                nearest_idx, nearest_pose = find_nearest_se3(current_pose, self.camera_poses_viewer)

                T_world_current = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3(event.client.camera.wxyz), event.client.camera.position
                )
                T_world_target = vtf.SE3.from_matrix(nearest_pose)

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(10):
                    T_world_set = T_world_current @ vtf.SE3.exp(
                        T_current_target.log() * j / 9.0
                    )

                    # We can atomically set the orientation and the position of the camera
                    # together to prevent jitter that might happen if one was set before the
                    # other.
                    with event.client.atomic():
                        event.client.camera.wxyz = T_world_set.rotation().wxyz
                        event.client.camera.position = T_world_set.translation()

                    event.client.flush()  # Optional!
                    time.sleep(1.0 / 20.0)
                
                frame_slider.value = nearest_idx

            @prev_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                frame_slider.value = (frame_slider.value - frame_step_slider.value) % len(self.camera_poses)
                    
            @next_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                frame_slider.value = (frame_slider.value + frame_step_slider.value) % len(self.camera_poses)
            
            @play_button.on_click
            def _(_) -> None:
                play_button.visible = False
                pause_button.visible = True
                snap_button.visible = False
                
                def play() -> None:
                    while not play_button.visible:
                        if len(self.camera_poses) > 0:
                            assert frame_slider is not None
                            frame_slider.value = (frame_slider.value + frame_step_slider.value) % len(self.camera_poses)
                        time.sleep(1.0 / 5)

                threading.Thread(target=play).start()
                
            @pause_button.on_click
            def _(_) -> None:
                play_button.visible = True
                pause_button.visible = False
                snap_button.visible = True
                

    def add_interactive_rendering_folder(self, server: viser.ViserServer):
        with server.add_gui_folder("Render"):
            self.max_res_when_static = server.add_gui_slider(
                "Max Res",
                min=128,
                max=3840,
                step=128,
                initial_value=1408,
            )
            self.max_res_when_static.on_update(self._handle_option_updated)
            self.jpeg_quality_when_static = server.add_gui_slider(
                "JPEG Quality",
                min=0,
                max=100,
                step=1,
                initial_value=100,
            )
            self.jpeg_quality_when_static.on_update(self._handle_option_updated)

            self.max_res_when_moving = server.add_gui_slider(
                "Max Res when Moving",
                min=128,
                max=3840,
                step=128,
                initial_value=768,
            )
            self.jpeg_quality_when_moving = server.add_gui_slider(
                "JPEG Quality when Moving",
                min=0,
                max=100,
                step=1,
                initial_value=50,
            )


    @torch.no_grad()
    def cutoff_gaussians(self):
        assert self.query_xyz is not None
        print(f"Perform cutoff with query {self.query_xyz}, threshold {self.cutoff_threshold}")
        
        D, I = self.kdtree_gaussians.query(self.query_xyz, k = 1)
        query_feature = self.gaussian_features[I] # (1, D)
        
        feature_dists = (self.gaussian_features - query_feature).norm(dim=-1) # (N, )
        inside_mask = feature_dists < self.cutoff_threshold # (N, )
        
        new_opacity = self.original_opacity.clone()
        new_opacity[~inside_mask] = new_opacity[~inside_mask] - 100
        
        self.model.gaussians._opacity = torch.nn.Parameter(new_opacity)
        if self.model_feat_pca is not None:
            self.model_feat_pca.gaussians._opacity = torch.nn.Parameter(new_opacity)
        
        
    @torch.no_grad()
    def recover_gaussians(self):
        self.model.gaussians._opacity = torch.nn.Parameter(self.original_opacity.clone())
        if self.model_feat_pca is not None:
            self.model_feat_pca.gaussians._opacity = torch.nn.Parameter(self.original_opacity.clone())


    def add_point_query_folder(self, server: viser.ViserServer):
        if hasattr(self, "point_query_folder") and self.point_query_folder is not None:
            self.point_query_folder.remove()

        with server.add_gui_folder("Point Query") as gui_folder:
            self.point_query_folder = gui_folder
            self.do_cutoff_gaussians = False
            self.cutoff_threshold = 0.6

            gaussian_xyz = self.model.gaussians.get_xyz

            # Add sliders to specify the query point
            coord_sliders = []
            for i, name in enumerate(['x', 'y', 'z']):
                v_min, v_max = gaussian_xyz[:, i].min().item(), gaussian_xyz[:, i].max().item()
                slider = server.add_gui_slider(
                    f"{name} coord",
                    min=v_min,
                    max=v_max,
                    step=0.005,
                    initial_value=(v_min + v_max) / 2,
                )
                coord_sliders.append(slider)
                
            # Visualize the query point and link it to the sliders
            self.query_xyz = np.asarray([slider.value for slider in coord_sliders])[None]
            self.query_colors = np.asarray([255, 0, 0])[None]
            self.query_pcd = server.add_point_cloud(
                "query_pcd",
                points=self.query_xyz,
                colors=self.query_colors,
                point_size=0.02,
            )
            def update_query_xyz(event: viser.GuiEvent):
                with self.server.atomic():
                    self.query_xyz = np.asarray([slider.value for slider in coord_sliders])[None]
                    self.query_pcd.remove()
                    self.query_pcd = server.add_point_cloud(
                        "query_pcd",
                        points=self.query_xyz,
                        colors=self.query_colors,
                        point_size=0.02,
                    )
                    # Cut-off Gaussians and Re-render
                    if self.do_cutoff_gaussians: self.cutoff_gaussians()
                    self._handle_option_updated(event) 
                    
            update_query_xyz(None)
            
            for slider in coord_sliders:
                slider.on_update(update_query_xyz)
                
            # Add another slide for threshold values 
            thresh_slider = server.add_gui_slider(
                "Threshold",
                min=0,
                max=5,
                step=0.01,
                initial_value=self.cutoff_threshold,
            )
            @thresh_slider.on_update
            def _(_) -> None:
                self.cutoff_threshold = thresh_slider.value
                # Cut-off Gaussians and Re-render
                with self.server.atomic():
                    if self.do_cutoff_gaussians: self.cutoff_gaussians()
                self._handle_option_updated(_) 
            
            # Add checkbox for whether to do thresholding
            cutoff_checkbox = server.add_gui_checkbox(
                "Cutoff Gaussians",
                initial_value=self.do_cutoff_gaussians,
                hint="Whether to cutoff the gaussians",
            )
            @cutoff_checkbox.on_update
            def _(_) -> None:
                self.do_cutoff_gaussians = cutoff_checkbox.value
                # Cut-off Gaussians and Re-render
                with self.server.atomic():
                    if self.do_cutoff_gaussians: self.cutoff_gaussians()
                if self.do_cutoff_gaussians is False:
                    self.recover_gaussians()
                self._handle_option_updated(_) 
                
            # A checkbox for visibility of the query point
            query_visibility_checkbox = server.add_gui_checkbox(
                "Show Query",
                initial_value=False,
                hint="Whether to show the query point",
            )
            self.query_pcd.visible = query_visibility_checkbox.value
            @query_visibility_checkbox.on_update
            def _(_) -> None:
                self.query_pcd.visible = query_visibility_checkbox.value
                self._handle_option_updated(_) 
                

    def start(self):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        server.configure_theme(
            control_layout="collapsible",
            show_logo=False,
        )
        # register hooks
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

        self.server = server

        tabs = server.add_gui_tab_group()

        with tabs.add_tab("General"):
            self.model_paths_dropdown = server.add_gui_dropdown(
                "Model Paths", tuple(self.model_names), 
                initial_value=self.model_names[0]
            )
            
            self.model_iters_drop_down = server.add_gui_dropdown(
                "Model Iterations", tuple(self.ckpt_subpaths),
                initial_value=self.ckpt_subpaths[-1]
            )
            
            # A button to set the current camera up direction to be the up direction
            reset_up_button = server.add_gui_button(
                "Reset up direction",
                icon=viser.Icon.ARROW_AUTOFIT_UP,
                hint="Set the current up direction as the up direction.",
            )

            @reset_up_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                event.client.camera.up_direction = vtf.SO3(event.client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
                
            # A button to set the camera up direction to be the gravity direction
            set_top_up_button = server.add_gui_button(
                "Align Aria gravity",
                icon=viser.Icon.ARROW_AUTOFIT_UP,
                hint="Align the camera up direction with the gravity direction.",
            )
            
            @set_top_up_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                event.client.camera.up_direction = self.aria_up_direction
                
            # A checkbox to lock aspect ratio to be 1
            lock_aspect_one_checkbox = server.add_gui_checkbox(
                "Lock aspect ratio to 1",
                initial_value=self.lock_aspect_one,
                hint="Lock the aspect ratio of rendering to 1.",
            )
            
            @lock_aspect_one_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                self.lock_aspect_one = lock_aspect_one_checkbox.value
                

            # add cameras
            if self.show_cameras is True:
                self.add_cameras_to_scene(server)

            # add interactive rendering options
            self.add_interactive_rendering_folder(server)
            

            with server.add_gui_folder("Model"):
                self.scaling_modifier = server.add_gui_slider(
                    "Scaling Modifier",
                    min=0.,
                    max=1.,
                    step=0.1,
                    initial_value=1.,
                )
                self.scaling_modifier.on_update(self._handle_option_updated)

                if self.model.get_max_sh_degree() > 0:
                    self.active_sh_degree_slider = server.add_gui_slider(
                        "Active SH Degree",
                        min=0,
                        max=self.model.get_max_sh_degree(),
                        step=1,
                        initial_value=self.model.get_max_sh_degree(),
                    )
                    self.active_sh_degree_slider.on_update(self._handle_activate_sh_degree_slider_updated)

                self.time_slider = server.add_gui_slider(
                    "Time",
                    min=0.,
                    max=1.,
                    step=0.01,
                    initial_value=0.,
                )
                self.time_slider.on_update(self._handle_option_updated)
                
                if self.add_feat_pca:
                    self.feat_pca_checkbox = server.add_gui_checkbox(
                        "Use PCA Visualized Features",
                        initial_value=False,
                        hint="Whether to use the PCA visualized features.",
                    )
                    @self.feat_pca_checkbox.on_update
                    def _(_) -> None:
                        with server.atomic():
                            self.using_pca = self.feat_pca_checkbox.value
                            for i in self.clients:
                                self.clients[i].model = self.model_feat_pca if self.using_pca else self.model
                            self._handle_option_updated(_)
                
            @self.model_paths_dropdown.on_update
            def _(_) -> None:
                model_path = self.get_current_model_path()
                with server.atomic():
                    # self.model_iters_drop_down.options = scan_avail_iters(model_path)
                    self.model_iters_drop_down.options = get_all_ckpt_subpaths(model_path)
                    self.model_iters_drop_down.value = self.model_iters_drop_down.options[-1]
                self.init_from_model_path(model_path, self.model_iters_drop_down.value)
                self.rerender_for_all_client()
                self.add_frame_slider_folder(server)
                self.add_point_query_folder(server)
                
            @self.model_iters_drop_down.on_update
            def _(_) -> None:
                model_path = self.get_current_model_path()
                self.init_from_model_path(model_path, self.model_iters_drop_down.value)
                self.rerender_for_all_client()
                self.add_point_query_folder(server)
                
            self.add_frame_slider_folder(server)
            self.add_point_query_folder(server)
            
            self.add_capture_save_folder(server)
            

        if self.show_edit_panel is True:
            with tabs.add_tab("Edit") as edit_tab:
                self.edit_panel = EditPanel(server, self, edit_tab)

        self.transform_panel: TransformPanel = None
        if self.enable_transform is True:
            with tabs.add_tab("Transform"):
                self.transform_panel = TransformPanel(server, self, self.loaded_model_count)

        if self.show_render_panel is True:
            with tabs.add_tab("Render"):
                populate_render_tab(
                    server,
                    self,
                    self.model_paths,
                    Path("./"),
                    orientation_transform=torch.linalg.inv(self.camera_transform).cpu().numpy(),
                    enable_transform=self.enable_transform,
                    background_color=self.background_color,
                    sh_degree=self.sh_degree,
                )

        while True:
            time.sleep(999)

    def _handle_appearance_embedding_slider_updated(self, event: viser.GuiEvent):
        """
        Change appearance group dropdown to "@Direct" on slider updated
        """

        if event.client is None:  # skip if not updated by client
            return
        self.appearance_group_dropdown.value = DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE
        self._handle_option_updated(event)

    def _handle_activate_sh_degree_slider_updated(self, _):
        self.model.set_active_sh_degree(self.active_sh_degree_slider.value)
        self._handle_option_updated(_)

    def get_appearance_id_value(self):
        """
        Return appearance id according to the slider and dropdown value
        """

        # no available appearance options, simply return zero
        if self.available_appearance_options is None:
            return (0, 0.)
        name = self.appearance_group_dropdown.value
        # if the value of dropdown is "@Direct", or not in available_appearance_options, return the slider's values
        if name == DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE or name not in self.available_appearance_options:
            return (self.appearance_id.value, self.normalized_appearance_id.value)
        # else return the values according to the dropdown
        return self.available_appearance_options[name]

    def _handel_appearance_group_dropdown_updated(self, event: viser.GuiEvent):
        """
        Update slider's values when dropdown updated
        """

        if event.client is None:  # skip if not updated by client
            return

        # get appearance ids according to the dropdown value
        appearance_id, normalized_appearance_id = self.available_appearance_options[self.appearance_group_dropdown.value]
        # update sliders
        self.appearance_id.value = appearance_id
        self.normalized_appearance_id.value = normalized_appearance_id
        # rerender
        self._handle_option_updated(event)

    def _handle_option_updated(self, _):
        """
        Simply push new render to all client
        """
        return self.rerender_for_all_client()

    def handle_option_updated(self, _):
        return self._handle_option_updated(_)

    def rerender_for_client(self, client_id: int):
        """
        Render for specific client
        """
        try:
            # switch to low resolution mode first, then notify the client to render
            self.clients[client_id].state = "low"
            self.clients[client_id].render_trigger.set()
        except:
            # ignore errors
            pass

    def rerender_for_all_client(self):
        for i in self.clients:
            self.rerender_for_client(i)

    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        """
        Create and start a thread for every new client
        """

        # create client thread
        client_thread = ClientThread(self, self.model, client)
        client_thread.start()
        # store this thread
        self.clients[client.client_id] = client_thread

    def _handle_client_disconnect(self, client: viser.ClientHandle):
        """
        Destroy client thread when client disconnected
        """

        try:
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as err:
            print(err)


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=str)
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--reorient", "-r", type=str, default="auto",
                        help="whether reorient the scene, available values: auto, enable, disable")
    parser.add_argument("--enable_transform", "--enable-transform",
                        action="store_true", default=False,
                        help="Enable transform options on Web UI. May consume more memory")
    parser.add_argument("--show_cameras", "--show-cameras",
                        action="store_true")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--feat_pca", action="store_true", 
                        help="If set, also provide PCA visualization for learned features. ")
    args = parser.parse_args()

    # create viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args)}
    viewer = Viewer(**viewer_init_args)

    # start viewer server
    viewer.start()

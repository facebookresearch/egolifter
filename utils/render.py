# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
from typing import Optional
import imageio
from matplotlib import pyplot as plt
import numpy as np
import torch
import open3d as o3d
import tqdm

from torch.utils.data import DataLoader

from gaussian_renderer.gsplat import render
from scene import GaussianModel, Scene
from scene.cameras import Camera
from natsort import natsorted
from moviepy.editor import ImageSequenceClip
from moviepy.editor import clips_array
import torchvision

from utils.constants import FOV_IN_RAD, MIMSAVE_ARGS
from utils.pca import FeatPCA


def render_set(
        model_path: str, 
        name: str, 
        scene: Scene, 
        loader: DataLoader,
        gaussians: GaussianModel, 
        args_pipe: argparse.Namespace, 
        background: torch.Tensor, 
        render_feature: bool,
        save_ext: str
    ):
    if len(loader) == 0:
        print(f"No {name} views found. ")
        return None
    
    iteration = scene.loaded_iter
    if iteration is None:
        iteration = "latest"
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_features_path = os.path.join(model_path, name, "ours_{}".format(iteration), "features")

    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(render_features_path, exist_ok=True)

    for idx, batch in enumerate(tqdm.tqdm(loader, desc="Rendering progress")):
        gt = batch["image"].cuda()
        view = scene.get_camera(batch["idx"].item(), subset=name)

        # First only render the image
        render_pkg = render(view, gaussians, args_pipe, background, render_feature = False)
        render_rgb = render_pkg["render"]

        render_rgb, gt = scene.postProcess(render_rgb, gt, view)
        
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # Instead of duplicate the images, create a soft link
        src = view.image_path
        gt_ext = src.split(".")[-1]
        dst = os.path.join(gts_path, '{0:05d}'.format(idx) + f".{gt_ext}")
        if not os.path.exists(dst):
            os.symlink(src, dst)
        
        torchvision.utils.save_image(render_rgb, os.path.join(render_path, '{0:05d}'.format(idx) + f".{save_ext}"))

        if render_feature:
            # Then change the size and render the feature
            view_mask = view.copy()
            view_mask.image_width = 512
            view_mask.image_height = 512

            render_pkg = render(view_mask, gaussians, args_pipe, background, render_feature = True)
            render_features = render_pkg['render_features']
            render_features = render_features.detach().cpu().numpy()
            np.save(os.path.join(render_features_path, '{0:05d}'.format(idx) + ".npy"), render_features)
            
    return concate_to_video(os.path.join(model_path, name, "ours_{}".format(iteration)), 20, False)

def concate_to_video(input_folder, fps, no_rot):
    gt_folder = os.path.join(input_folder, "gt")
    gt_images = [os.path.join(gt_folder, img) for img in os.listdir(gt_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    gt_images = natsorted(gt_images)

    render_folder = os.path.join(input_folder, "renders")
    render_images = [os.path.join(render_folder, img) for img in os.listdir(render_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    render_images = natsorted(render_images)

    assert len(gt_images) == len(render_images), f"Number of images in gt {len(gt_images)} and render {len(render_images)} folders are not equal."

    # Create a moviepy video clip
    gt_clip = ImageSequenceClip(gt_images, fps=fps)
    render_clip = ImageSequenceClip(render_images, fps=fps)

    # Rotate the aria images to upright. 
    if not no_rot:
        gt_clip = gt_clip.rotate(-90)
        render_clip = render_clip.rotate(-90)

    clip = clips_array([[gt_clip, render_clip]])
    clip = clip.resize(width = min(clip.w, 3000))

    # Write the video file
    output_video_path = os.path.join(input_folder, "gt_render.mp4")
    clip.write_videofile(output_video_path, fps=fps)
    
    return output_video_path

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize(x):
    return x / np.linalg.norm(x)

def look_at(cam_center, obj_center, up):
    """Generate a look-at transformation matrix."""
    f = (obj_center - cam_center) / np.linalg.norm(obj_center - cam_center)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    u /= np.linalg.norm(u)

    m = np.eye(4)
    m[:3, :3] = [r, -u, f]
    m[:3, :3] = m[:3, :3].T
    m[:3, 3] = cam_center
    return m

def spiral_camera_trajectory(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        # render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        mat = viewmatrix(z, up, c)
        mat = np.concatenate([mat, np.array([[0,0,0,1]])], axis=0)
        render_poses.append(mat)

    return render_poses

def rotating_camera_trajectory(
    center: np.ndarray, # The rotation center and look-at point
    up_vec: np.ndarray, # Up vector of the object
    lifting_angle: float, # The angle to lift the camera
    distance: float, # The distance of the camera from the object
    rots=1, N=60, # The number of rotations and camera poses
):
    # Generate the camera positions
    angles = np.linspace(0, 2 * np.pi * rots, N)
    cam_positions = np.zeros((N, 3))
    for i, theta in enumerate(angles):
        cam_x = distance * np.cos(theta) * np.cos(np.radians(lifting_angle))
        cam_y = distance * np.sin(theta) * np.cos(np.radians(lifting_angle))
        cam_z = distance * np.sin(np.radians(lifting_angle))
        cam_positions[i] = [cam_x, cam_y, cam_z]

    # Calculate camera poses
    render_poses = np.zeros((N, 4, 4))
    for i, pos in enumerate(cam_positions):
        camera_matrix = look_at(pos + center, center, up_vec)
        render_poses[i] = camera_matrix

    return render_poses

def get_render_path_rotate(points, up_vec, lifting_angle=0.0, rots=1, N=60, dist_factor=1.5):
    '''
    Generate a list of camera poses that rotate the camera around the object. 
    The camera is placed at the distance of the diagonal of the bounding box of the object, scaled by dist_factor.
    The up direction of camera aligns with the up_vec.
    The camera is always looking at the center of the object. 

    Args:
        points: (N, 3) numpy array. The point cloud of the object.
        up_vec: (3,) numpy array. The up direction of the camera.
        lifting_angle: float. The angle in degree to lift the camera from the center of the object.
        rots: int. The number of circles rotating around the object.
        N: int. The number of camera poses to generate. N poses are even distributed along the rotation.
        dist_factor: float. The factor to scale the distance of the camera from the object. The distance is the diagonal of the bounding box of the object.

    Returns:
        render_poses: (N, 4, 4) numpy array. The camera poses.
    '''

    # Calculate the center of the object
    center = np.mean(points, axis=0)

    # Calculate the bounding box diagonal length
    max_bound = np.max(points, axis=0)
    min_bound = np.min(points, axis=0)
    diagonal = np.linalg.norm(max_bound - min_bound)
    max_dist = np.max(np.linalg.norm(points - center, axis=1))

    # Calculate the distance of the camera from the object's center
    distance = diagonal * dist_factor
    distance = max(distance, max_dist + 0.2) # At least 25 cm away to avoid culling artifacts

    render_poses = rotating_camera_trajectory(
        center, 
        up_vec,
        lifting_angle,
        distance,
        rots,
        N,
    )

    return render_poses

def get_render_poses_spiral(
    xyz: np.ndarray,
    camera_extrinsic: np.ndarray,
    rad_scale: float = 0.3,
    N_views: int = 150,
    N_rots: int = 1,
):
    c2w = np.linalg.inv(camera_extrinsic)
    up = c2w[:3, 1]
    
    obj_center = np.mean(xyz, axis=0)
    obj_extent = np.std(xyz @ c2w[:3, :3].T, axis=0)
    rads = obj_extent
    # rads = c2w[:3, :3] @ rads
    rads = rads * rad_scale
    focal = np.linalg.norm(obj_center - c2w[:3, 3]) * 50
    render_poses = spiral_camera_trajectory(
        c2w, up, rads, focal=focal, zrate=.5, rots=N_rots, N=N_views
    )
    return render_poses

@torch.no_grad()
def render_gaussians_camera_poses(
    gaussians: GaussianModel,
    train_view: Camera,
    args_pipe: argparse.Namespace,
    render_poses: np.ndarray,
    uniform_color = None,
    background = torch.tensor((1.0, 1.0, 1.0)),
    check_cameras = False,
    save_path: str = None,
    fov_in_rad: float = None, # If None, do not change fov from the training
):
    xyz = gaussians.get_xyz.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if uniform_color is not None:
        pcd.paint_uniform_color(uniform_color)

    if check_cameras:
        visualize_pcd_with_cameras([pcd], render_poses)

    rendered_frames = []
    for i in range(len(render_poses)):
        camera_pose = render_poses[i]
        camera_pose = np.linalg.inv(camera_pose)
        R = camera_pose[:3,:3]
        R = R.T
        t = camera_pose[:3,3]

        # Set the fov to 90 degrees
        if fov_in_rad is None:
            view = Camera(
                colmap_id=0, uid=0, R=R, T=t,
                FoVx=train_view.FoVx, FoVy=train_view.FoVy,
                image_width=train_view.image_width, image_height=train_view.image_height,
                image_name="", image_path="",
            )
        else:
            view = Camera(
                colmap_id=0, uid=0, R=R, T=t,
                FoVx=fov_in_rad, FoVy=fov_in_rad,
                image_width=train_view.image_width, image_height=train_view.image_height,
                image_name="", image_path="",
            )

        render_pkg = render(view, gaussians, args_pipe, background, render_feature=False)
        render_rgb = render_pkg["render"]
        render_rgb = render_rgb.detach().cpu().numpy().transpose(1, 2, 0)
        render_rgb = render_rgb.clip(0.0, 1.0)
        render_rgb = (render_rgb * 255).astype(np.uint8)

        rendered_frames.append(render_rgb)
    
    if save_path is None:
        return rendered_frames
    else:
        # Save and return an empty list to save memory usage
        imageio.mimsave(save_path, rendered_frames, **MIMSAVE_ARGS)
        # # Save the first frame as well for debugging
        # imageio.imwrite(save_path.replace(".mp4", ".jpg"), rendered_frames[0])
        return []
    
@torch.no_grad()
def render_gaussians_featpca_poses(
    gaussians: GaussianModel,
    train_view: Camera,
    args_pipe: argparse.Namespace,
    render_poses: np.ndarray,
    pca: FeatPCA,
    background = torch.tensor((1.0, 1.0, 1.0)),
    check_cameras = False,
    save_path: str = None,
    fov_in_rad: float = None, # If None, do not change fov from the training
):
    if check_cameras:
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        visualize_pcd_with_cameras([pcd], render_poses)

    rendered_frames = []
    for i in range(len(render_poses)):
        camera_pose = render_poses[i]
        camera_pose = np.linalg.inv(camera_pose)
        R = camera_pose[:3,:3]
        R = R.T
        t = camera_pose[:3,3]

        # Set the fov to 90 degrees
        if fov_in_rad is None:
            view = Camera(
                colmap_id=0, uid=0, R=R, T=t,
                FoVx=train_view.FoVx, FoVy=train_view.FoVy,
                image_width=train_view.image_width, image_height=train_view.image_height,
                image_name="", image_path="",
            )
        else:
            view = Camera(
                colmap_id=0, uid=0, R=R, T=t,
                FoVx=fov_in_rad, FoVy=fov_in_rad,
                image_width=train_view.image_width, image_height=train_view.image_height,
                image_name="", image_path="",
            )

        render_pkg = render(view, gaussians, args_pipe, background, render_feature=True)
        render_feat = render_pkg["render_features"].permute(1, 2, 0).cpu().numpy() # (H, W, D)
        feat_shape = render_feat.shape[:2]
        render_feat = render_feat.reshape(-1, render_feat.shape[-1]) # (N, D)
        render_pca = pca.transform(render_feat) # (N, 3)
        render_pca = render_pca.reshape(feat_shape[0], feat_shape[1], 3) # (H, W, 3)
        render_pca = (render_pca * 255).astype(np.uint8)
        
        # render_rgb = render_pkg["render"]
        # render_rgb = render_rgb.detach().cpu().numpy().transpose(1, 2, 0)
        # render_rgb = render_rgb.clip(0.0, 1.0)
        # render_rgb = (render_rgb * 255).astype(np.uint8)

        rendered_frames.append(render_pca)
    
    if save_path is None:
        return rendered_frames
    else:
        # Save and return an empty list to save memory usage
        imageio.mimsave(save_path, rendered_frames, **MIMSAVE_ARGS)
        # # Save the first frame as well for debugging
        # imageio.imwrite(save_path.replace(".mp4", ".jpg"), rendered_frames[0])
        return []
    
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

def visualize_pcd_with_cameras(
    geometries, camera_poses,
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2560, height=1440)
    for g in geometries:
        vis.add_geometry(g)
    cmap = plt.get_cmap('jet')
    for i, pose in enumerate(camera_poses):
        color = cmap(float(i) / len(camera_poses))
        frustum = better_camera_frustum(pose, 300, 300, scale=0.15, color=color[:3])
        vis.add_geometry(frustum)
    # Draw a coordinate system at the origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Load or generate a test point cloud
    pcd = o3d.geometry.PointCloud()
    points = np.random.rand(1000, 3)  # Random points for test purposes. Replace with actual point cloud if available.
    pcd.points = o3d.utility.Vector3dVector(points)
    geometries = [pcd]

    # Define the up vector for the camera
    up_vec = np.array([0, 0, 1])  # This points upward in the Y-direction

    # Call the function to generate camera poses
    camera_poses = get_render_path_rotate(
        points, 
        up_vec, 
        lifting_angle = 10.0, 
        dist_factor = 1.0,
    )

    # Visualize the point cloud and camera poses
    visualize_pcd_with_cameras(geometries, camera_poses)
    
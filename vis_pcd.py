from collections import Counter
from datetime import datetime
import copy
import os, sys, argparse
import re
from glob import glob
from pathlib import Path
import distinctipy
import imageio

from natsort import natsorted

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
import sklearn
import torch
import yaml
import clip

# https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
import faiss
import pytorch_lightning as L


from scene import GaussianModel, Scene
from scene.cameras import Camera
from utils.o3d import (
    pcd_mask_color,
    visualize_pcd_with_cameras,
    o3d_visualize_capture_trajectory, 
    get_bbox_geometries_by_label, 
    visualize_get_extrinsic_click,
    interactive_merge_cluster,
    visualize_get_extrinsic,
)
from utils.render import (
    get_render_path_rotate,
    get_render_poses_spiral,
    rotating_camera_trajectory,
    render_gaussians_camera_poses,
    render_gaussians_featpca_poses
)
from utils.gaussians import blend_gaussian_with_color, convert_gaussian_grayscale
from utils.gaussian_cluster import gaussian_cluster_instance
from utils.constants import UP_VEC, FOV_IN_RAD, FOV_IN_DEG, MIMSAVE_ARGS
from utils.torch_utils import cosine_similarity_batch
from utils.routines import load_from_model_path
from utils.pca import compute_feat_pca_from_renders


LSEG_CKPT_PATH = "/home/qgu/Downloads/lseg_minimal_e200.ckpt"

def picked_xyz_to_cluster(
    xyz: np.ndarray,
    point_cluster_ids: np.ndarray,
    picked_xyz: np.ndarray,
    nn_dist_thresh: float = 0.05,
    k: int = 5,
):
    D = xyz.shape[1]
    index = faiss.IndexFlatL2(D)
    index.add(xyz.astype(np.float32))

    dists, indices = index.search(picked_xyz.astype(np.float32), k)
    picked_cluster = []
    for i in range(len(picked_xyz)):
        nn_indices = indices[i][dists[i] < nn_dist_thresh]
        nn_cluster_ids = point_cluster_ids[nn_indices]
        if len(nn_indices) == 0 or (nn_cluster_ids == -1).all():
            picked_cluster.append(-1)
        else:
            nn_cluster_ids = nn_cluster_ids[nn_cluster_ids != -1]
            cluster_id, count = Counter(nn_cluster_ids).most_common(1)[0]
            picked_cluster.append(cluster_id)

    picked_cluster = np.asarray(picked_cluster)
    return picked_cluster
    



    
@torch.no_grad()
def compute_color_by_text(
    text_query: str|list, 
    features : np.ndarray, 
    clip_text_encoder: torch.nn.Module, 
    pca: sklearn.decomposition.PCA,
):
    if isinstance(text_query, list):
        text_prompts = text_query
    else:
        text_prompts = text_query.split(",")
        

    if len(text_prompts) == 1:
        prompt = clip.tokenize(text_query)
        # Color the point cloud by the similarity
        prompt = prompt.cuda()
        text_feat = clip_text_encoder(prompt)
        text_feat_norm = torch.nn.functional.normalize(text_feat, dim=-1)
        text_feat_pca = pca.transform(text_feat_norm.cpu().numpy())
        text_feat_pca = torch.from_numpy(text_feat_pca)

        point_feat_norm = torch.nn.functional.normalize(torch.tensor(features), dim=-1)

        # print(text_feat_pca.shape, point_feat_norm.shape)
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_similarity(
            point_feat_norm, text_feat_pca
        )

        similarity = similarity.detach().cpu().numpy()
        cmap = plt.get_cmap("viridis")
        colors = cmap(similarity)
        colors = colors[:, :3]
        return colors, None, None, classes
    else:
        # Color the point cloud by their most similar class
        text_prompts.append("other")
        text_feats = []
        for t in text_prompts:
            t = t.strip()
            prompt = clip.tokenize(t)
            prompt = prompt.cuda()
            text_feat = clip_text_encoder(prompt)
            text_feat_norm = torch.nn.functional.normalize(text_feat, dim=-1)
            text_feat_pca = pca.transform(text_feat_norm.cpu().numpy())
            text_feat_pca = torch.from_numpy(text_feat_pca)
            text_feats.append(text_feat_pca)

        text_feats = torch.cat(text_feats, dim=0).float()
        similarities = cosine_similarity_batch(
            torch.tensor(features), text_feats
        )

        classes = similarities.argmax(dim=1)
        palette = distinctipy.get_colors(len(text_prompts), rng=0)
        palette[-1] = [0, 0, 0]
        colors = np.array([palette[c] for c in classes])
        return colors, text_prompts, palette, classes




def render_clusters(
    gaussians: GaussianModel,
    train_view: Camera,
    args_pipe: argparse.Namespace,
    render_poses_scene: np.ndarray,
    background = torch.tensor((1.0, 1.0, 1.0)),
    picked_cluster = np.ndarray,
    point_labels_instance = np.ndarray,
    save_folder: str = None,
    render_fg: bool = True,
    render_fg_rotate_only: bool = False,
):
    os.makedirs(save_folder, exist_ok=True)

    # Render the full scene
    print("Rendering the full scene...")
    rendered_frames_full = render_gaussians_camera_poses(
        gaussians,
        train_view,
        args_pipe,
        render_poses = render_poses_scene,
        background = background,
        save_path = None if save_folder is None else os.path.join(save_folder, "render_full.mp4")
    )

    # Render the segmented objects individually
    rendered_frames_fg = []
    list_rendered_frames_fg_rotate = []

    if render_fg:
        for cluster_id in picked_cluster:
            print("Rendering foreground...")
            gaussians_cluster = copy.deepcopy(gaussians)
            mask_cluster = point_labels_instance == cluster_id
            gaussians_cluster.mask_points(mask_cluster)

            # Get a rotating render of the object
            xyz_cluster = gaussians_cluster.get_xyz.detach().cpu().numpy()
            render_poses_object = get_render_path_rotate(
                points = xyz_cluster,
                up_vec = UP_VEC, # gravity is along negative z axis
                lifting_angle = 30.0,
                rots = 1,
                N = 120,
                dist_factor = 0.7,
            )
            rendered_frames_cluster_rotate = render_gaussians_camera_poses(
                gaussians_cluster,
                train_view,
                args_pipe,
                render_poses = render_poses_object,
                background = background,
                save_path = None if save_folder is None else os.path.join(save_folder, f"render_fg_rotate_{cluster_id}.mp4"),
            )
            list_rendered_frames_fg_rotate.append(rendered_frames_cluster_rotate)
            
            # Then get a spiral render of the object
            if not render_fg_rotate_only:
                rendered_frames_cluster_spiral = render_gaussians_camera_poses(
                    gaussians_cluster,
                    train_view,
                    args_pipe,
                    render_poses = render_poses_scene,
                    background = background,
                    save_path = None if save_folder is None else os.path.join(save_folder, f"render_fg_spiral_{cluster_id}.mp4"),
                )
                # Repeat the full scene before rendering each individual object
                rendered_frames_fg += rendered_frames_full + rendered_frames_cluster_spiral

    # Render the scene with the objects removed
    gaussians_bg = copy.deepcopy(gaussians)
    mask_fg = np.isin(point_labels_instance, picked_cluster)
    mask_bg = np.logical_not(mask_fg)
    gaussians_bg.mask_points(mask_bg)
    print("Rendering background...")
    rendered_frames_bg = render_gaussians_camera_poses(
        gaussians_bg,
        train_view,
        args_pipe,
        render_poses = render_poses_scene,
        background = background,
        save_path = None if save_folder is None else os.path.join(save_folder, "render_bg.mp4"),
    )
    # Repeat the full scene again
    rendered_frames_bg = rendered_frames_full + rendered_frames_bg

    return rendered_frames_bg, rendered_frames_fg, list_rendered_frames_fg_rotate

def main(args: argparse.Namespace):
    assert args.input.exists(), f"{args.input} does not exist"
    
    model, scene, cfg = load_from_model_path(
        args.input, source_path=args.data, simple_scene=True)

    gaussians = model.gaussians
    args_pipe = cfg.pipe
    background = model.background
    if args.feat_dim is None:
        args.feat_dim = cfg.lift.contr_dim
    
    L.seed_everything(0)
    
    train_view = scene.getTrainCameras()[0].copy()

    xyz = gaussians.get_xyz
    xyz = xyz.detach().cpu().numpy()

    opacity = gaussians.get_opacity
    opacity = opacity.detach().cpu().numpy()
    opacity = opacity.reshape(-1)

    features = gaussians.get_features_extra
    features = features.detach().cpu().numpy()

    gaussians_init = GaussianModel(sh_degree=3, dim_extra=cfg.lift.contr_dim)
    gaussians_init.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)

    if args.render_resolution is not None:
        if len(args.render_resolution) == 1:
            render_width = args.render_resolution[0]
            render_height = args.render_resolution[0]
        elif len(args.render_resolution) == 2:
            render_width = args.render_resolution[0]
            render_height = args.render_resolution[1]
        else:
            raise ValueError(f"Unknown render_resolution: {args.render_resolution}")
    train_view.set_image_size(render_width, render_height)
    print(f"Rendering resolution: {train_view.image_width} x {train_view.image_height}")
    train_view.set_fov(FOV_IN_RAD, FOV_IN_RAD)

    # Run clustering algorithm or load the clustering results from file
    if args.load_cluster_path is None:
        point_labels_instance, point_labels_cluster_init, n_clusters, transparent_mask, cluster_noise_mask = gaussian_cluster_instance(
            args,
            features,
            xyz,
            covariance = gaussians.get_covariance_matrix().detach().cpu().numpy(),
            opacity = opacity,
        )
    else:
        point_labels_instance = np.load(args.load_cluster_path)
        n_clusters = point_labels_instance.max() + 1
        transparent_mask_path = os.path.join(os.path.dirname(args.load_cluster_path), "transparent_mask.npy")
        cluster_noise_mask_path = os.path.join(os.path.dirname(args.load_cluster_path), "cluster_noise_mask.npy")
        transparent_mask = np.load(transparent_mask_path) if os.path.exists(transparent_mask_path) else None
        cluster_noise_mask = np.load(cluster_noise_mask_path) if os.path.exists(cluster_noise_mask_path) else None

    # Color palette for the clusters
    palette = distinctipy.get_colors(n_clusters + 1, rng=0)
    palette[-1] = (0.6, 0.6, 0.6)
    palette = np.array(palette)

    # If needed, merge clusters interactively
    if args.pick_merge:
        point_labels_instance = interactive_merge_cluster(
            xyz,
            point_labels_instance,
            palette,
        )


    # Start saving results from this point
    save_folder = args.save_folder
    if save_folder is None:
        if args.save_suffix is None:
            save_folder = args.input / "vis_pcd"
        else:
            save_folder = args.input / f"vis_pcd-{args.save_suffix}"
    os.makedirs(save_folder, exist_ok=True)

    # Save information that can be saved at this point
    with open(os.path.join(save_folder, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    np.save(os.path.join(save_folder, "point_labels_instance.npy"), point_labels_instance)
    if transparent_mask is not None: np.save(os.path.join(save_folder, "transparent_mask.npy"), transparent_mask)
    if cluster_noise_mask is not None: np.save(os.path.join(save_folder, "cluster_noise_mask.npy"), cluster_noise_mask)
    

    # Visualize and pick foreground objects of interest
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    pcd_all.colors = o3d.utility.Vector3dVector(palette[point_labels_instance])


    # Pick clusters by either clicking points or loading from a given path
    picked_indices = None
    picked_xyz = None
    picked_cluster = None
    camera_extrinsic = None

    if args.load_picked_indices_path is None:
        print("Visualize the point cloud, colored by the clustering results. ")
        print("Please pick the cluster you want to render. ")
        camera_extrinsic, picked_indices = visualize_get_extrinsic_click(
            pcd_all,
            window_width=1920, 
            window_height=1920,
            fov = FOV_IN_DEG,
        )
        if len(picked_indices) == 0:
            if args.pick_merge:
                picked_indices = [0]
            else:
                print("No cluster selected, Exiting...")
                exit(0)
        picked_indices = np.array(picked_indices)
        picked_cluster = point_labels_instance[picked_indices]
        picked_xyz = xyz[picked_indices]
    else:
        if args.load_picked_indices_path.endswith("picked_xyz.npy"):
            print("Loading the picked xyz coordinates from", args.load_picked_indices_path)
            picked_xyz = np.load(args.load_picked_indices_path)
            picked_cluster = picked_xyz_to_cluster(
                xyz,
                point_labels_instance,
                picked_xyz,
                nn_dist_thresh=0.05,
                k=5,
            )
            print("Converted the picked xyz coordinates to cluster id :", picked_cluster)
        else:
            print("Loading the picked indices from", args.load_picked_indices_path)
            picked_indices = np.load(args.load_picked_indices_path)

            # Get the selected cluster
            picked_cluster = point_labels_instance[picked_indices]
            picked_xyz = xyz[picked_indices]

    picked_cluster = [c for c in picked_cluster if c != -1]
    picked_cluster = np.unique(picked_cluster)
    print("Selected cluster:", picked_cluster)

    if picked_indices is not None: np.save(os.path.join(save_folder, "picked_indices.npy"), picked_indices)
    if picked_cluster is not None: np.save(os.path.join(save_folder, "picked_cluster.npy"), picked_cluster)
    if picked_xyz is not None: np.save(os.path.join(save_folder, "picked_xyz.npy"), picked_xyz)
    
    bbox_geometries = get_bbox_geometries_by_label(
        xyz,
        point_labels_instance,
        picked_cluster, 
        palette,
        denoise_eps=args.bbox_denoise_eps,
    )

    # Get the points belonging to the selected cluster
    mask_fg = np.isin(point_labels_instance, picked_cluster)
    pcd_picked_cluster = o3d.geometry.PointCloud()
    pcd_picked_cluster.points = o3d.utility.Vector3dVector(xyz[mask_fg])
    pcd_picked_cluster.colors = o3d.utility.Vector3dVector(
        np.asarray(palette[point_labels_instance[mask_fg]])
    )
    pcd_all_bggray = pcd_mask_color(pcd_all, ~mask_fg, [0.6, 0.6, 0.6])
    

    # Generate the camera poses for visualizating the entire scene
    if args.load_render_poses_path is not None:
        print("Loading render_poses_scene from", args.load_render_poses_path)
        render_poses_scene = np.load(args.load_render_poses_path)
    else:
        print("Visualize the selected clusters (foreground object). ")
        print("Adjust the camera pose, which will be used for rendering poses. Press Q when finished. ")
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        camera_extrinsic = visualize_get_extrinsic(
            [pcd_all_bggray, coordinate_frame] + bbox_geometries,
            window_width=train_view.image_width, 
            window_height=train_view.image_height,
            init_extrinsic=camera_extrinsic,
            fov=FOV_IN_DEG,
        )
        
        if args.scene_camera_type == "spiral":
            render_poses_scene = get_render_poses_spiral(
                xyz,
                camera_extrinsic,
                rad_scale = args.spin_rad_scale,
            )
        elif args.scene_camera_type == "rotate":
            center = xyz[mask_fg].mean(0)

            # Compute the lifting angle by the camera_extrinsic and the center
            camera_pose = np.linalg.inv(camera_extrinsic)
            center_cam_vec =  camera_pose[:3, 3] - center
            distance = np.linalg.norm(center_cam_vec)
            center_cam_vec = center_cam_vec / distance
            lifting_angle = np.arcsin(center_cam_vec.dot(UP_VEC)) / np.pi * 180.0
            render_poses_scene = rotating_camera_trajectory(
                center,
                up_vec = UP_VEC,
                lifting_angle = lifting_angle,
                distance = distance,
                rots = 1,
                N = 300,
            )
        else:
            raise ValueError(f"Unknown scene_camera_type: {args.scene_camera_type}")

    np.save(os.path.join(save_folder, "render_poses_scene.npy"), render_poses_scene)

    if args.debug:
        # Viusalize the camera poses for debugging
        visualize_pcd_with_cameras(
            [pcd_picked_cluster], render_poses_scene,
            window_height=train_view.image_height,
            window_width=train_view.image_width,
            fov=FOV_IN_DEG,
        )

    render_extrinsics_scene = [np.linalg.inv(p) for p in render_poses_scene]

    if args.render_point_clouds:
        print("Render the Gaussian point cloud using Open3D. ")
        # Clustered and colored point cloud
        o3d_frames = o3d_visualize_capture_trajectory(
            [pcd_all],
            camera_extrinsics=render_extrinsics_scene, # convert poses to extrinsics
            window_width=train_view.image_width,
            window_height=train_view.image_height,
            field_of_view=FOV_IN_DEG,
        )
        imageio.mimsave(f"{save_folder}/render_pcd_all.mp4", o3d_frames, **MIMSAVE_ARGS)
        # save the first frame as well for debugging
        imageio.imwrite(f"{save_folder}/render_pcd_all.jpg", o3d_frames[0])
        
        # Clustered and colored point cloud, with background colored in gray
        o3d_frames = o3d_visualize_capture_trajectory(
            [pcd_all_bggray],
            camera_extrinsics=render_extrinsics_scene, # convert poses to extrinsics
            window_width=train_view.image_width,
            window_height=train_view.image_height,
            field_of_view=FOV_IN_DEG,
        )
        imageio.mimsave(f"{save_folder}/render_pcd_final.mp4", o3d_frames, **MIMSAVE_ARGS)

        # Clustered and colored point cloud, with bboxes
        o3d_frames = o3d_visualize_capture_trajectory(
            [pcd_all_bggray] + bbox_geometries,
            camera_extrinsics=render_extrinsics_scene, # convert poses to extrinsics
            window_width=train_view.image_width,
            window_height=train_view.image_height,
            field_of_view=FOV_IN_DEG,
        )
        imageio.mimsave(f"{save_folder}/render_pcd_final_bbox.mp4", o3d_frames, **MIMSAVE_ARGS)

        # initial semi-dense points
        pcd_init = o3d.geometry.PointCloud()
        pcd_init.points = o3d.utility.Vector3dVector(gaussians_init.get_xyz.detach().cpu().numpy())
        pcd_init.paint_uniform_color([0.6, 0.6, 0.6])
        o3d_frames = o3d_visualize_capture_trajectory(
            [pcd_init],
            camera_extrinsics=render_extrinsics_scene, # convert poses to extrinsics
            window_width=train_view.image_width,
            window_height=train_view.image_height,
            field_of_view=FOV_IN_DEG,
        )
        imageio.mimsave(f"{save_folder}/render_pcd_init.mp4", o3d_frames, **MIMSAVE_ARGS)

        # semi-dense points with bbox
        o3d_frames = o3d_visualize_capture_trajectory(
            [pcd_init] + bbox_geometries,
            camera_extrinsics=render_extrinsics_scene, # convert poses to extrinsics
            window_width=train_view.image_width,
            window_height=train_view.image_height,
            field_of_view=FOV_IN_DEG,
        )
        imageio.mimsave(f"{save_folder}/render_pcd_init_bbox.mp4", o3d_frames, **MIMSAVE_ARGS)


    # Render the scene with PCA colorization
    pca = compute_feat_pca_from_renders(scene, "train", [model])[0]
    render_gaussians_featpca_poses(
        gaussians,
        train_view,
        args_pipe,
        render_poses=render_poses_scene,
        pca=pca,
        background=background,
        save_path=os.path.join(save_folder, "render_pca.mp4"),
        fov_in_rad=FOV_IN_RAD,
    )


    # Render the scene with Gaussian Splatting - RGB
    print("Rendering the clusters using Gaussian Splatting...")
    print("Result frames will be saved to ", save_folder)
    render_clusters(
        gaussians,
        train_view,
        args_pipe,
        background = background,
        render_poses_scene = render_poses_scene,
        picked_cluster = picked_cluster,
        point_labels_instance = point_labels_instance,
        save_folder=save_folder,
        render_fg=args.render_each_clicked, 
        render_fg_rotate_only=args.render_clicked_rotate_only,
    )
    

    # Render the scene with Gaussian Splatting - Gray scale blended with cluster color
    convert_gaussian_grayscale(gaussians)
    blend_gaussian_with_color(
        gaussians, 
        palette[point_labels_instance],
        mask = point_labels_instance != -1,
        alpha = 0.75
    )
    render_gaussians_camera_poses(
        gaussians,
        train_view,
        args_pipe,
        render_poses = render_poses_scene,
        background = background,
        save_path = None if save_folder is None else os.path.join(save_folder, "render_full_blend.mp4"),
        fov_in_rad=FOV_IN_RAD,
    )

    

    # Make a video for each of the folders saving the rendered frames
    print("Converting frames to videos, if there are any...")
    folders = glob(os.path.join(save_folder, "render_*"))
    folders = [f for f in folders if os.path.isdir(f)]
    for folder in folders:
        print("Processing folder", folder)
        images = glob(os.path.join(folder, "*.jpg"))
        images = natsorted(images)
        images = [imageio.imread(img) for img in images]
        imageio.mimsave(
            f"{folder}.mp4", 
            images, **MIMSAVE_ARGS
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Path to the trained point cloud")
    parser.add_argument("-d", "--data", type=Path, default=None,
                        help="Path to the original data (Needed for the PCA model file)")
    parser.add_argument("-s", "--save_folder", type=Path, default=None,
                        help="Path to the folder to save the results")
    parser.add_argument("--debug", action="store_true",
                        help="If set, run this script with in debugging mode. ")
    
    parser.add_argument("--save_suffix", type=str, default=None,
                        help="The suffix for the saved files. ")

    parser.add_argument("--feat_dim", type=int, default=None,
                        help="Feature dimension. If None, set according to cfg")
    parser.add_argument("--seman_dim", type=int, default=0,
                        help="The feature dimension of the semantic part. ")
    
    parser.add_argument("--text_query", type=str, default="table,door,light,cabinet", 
                        help="The text prompt for coloring the point cloud")
    
    parser.add_argument("--opacity_thresh", type=float, default=0.5,
                        help="Gaussians with opacity below this threshold will be filtered out for visualization.")
    parser.add_argument("--spin_rad_scale", type=float, default=0.25, 
                        help="The scale of the spin radius for the rendering. ")

    parser.add_argument('--num_clusters', type=int, default=10,
                        help='Number of clusters for clustering. ')
    parser.add_argument("--spatial_weight", type=float, default=0.0,
                        help="the weight of the spatial position for clustering. ")
    parser.add_argument("--load_cluster_path", type=str, default=None,
                        help="If set, the clustering results will be loaded from this path. ")
    parser.add_argument("--load_picked_indices_path", type=str, default=None,
                        help="if set, the picked point indices will be loaded from this path. ")
    parser.add_argument("--load_render_poses_path", type=str, default=None,
                        help="if set, the render_poses_scene will be loaded from this path.")

    parser.add_argument("--pick_merge", action="store_true",
                        help="If set, interactive_merge_cluster will be run. ")
    parser.add_argument("--scene_camera_type", type=str, 
                        choices=['spiral', 'rotate'], default="rotate",
                        help="The type of camera trajectory to use for rendering the scene. ")
    
    parser.add_argument("--render_each_clicked", action="store_true")
    parser.add_argument("--render_point_clouds", action="store_true")
    parser.add_argument("--render_clicked_rotate_only", action="store_true")

    parser.add_argument("--bbox_denoise_eps", type=float, default=0.05)

    parser.add_argument("--render_resolution", type=int, nargs="*", default=None, 
                        help="The resolution of the rendered image (width height). If None, use that from training. ")
    
    parser.add_argument("--clusterer", type=str, default="kmeans", 
                        help="The clusterer to use. ")
    parser.add_argument("--reassign_dist_thresh", type=float, default=None,
                        help="If L2 distance from nearest Gaussian is larger than this, still consider it as noise. If None, do not check. ")
    parser.add_argument("--reassign_bhatta_dist_thresh", type=float, default=None,
                        help="If Bhattacharyya distance from nearest Gaussian is larger than this, still consider it as noise. If None, do not check. ")
    
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=15)
    parser.add_argument("--hdbscan_min_samples", type=int, default=1)
    parser.add_argument("--hdbcsan_epsilon", type=float, default=0.7)

    parser.add_argument("--denoise_dbscan_eps", type=float, default=-1,
                        help="The eps parameter for DBSCAN. ")
    parser.add_argument("--denoise_dbscan_min_points", type=int, default=50,
                        help="The min_points parameter for DBSCAN. ")
    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    main(args)
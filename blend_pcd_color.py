from datetime import datetime
from collections import Counter
import copy
import os, sys, argparse
import re
import pickle
from glob import glob
from pathlib import Path
import distinctipy

from natsort import natsorted

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import distinctipy

sys.path.append("/home/qgu/local/gaussian-splatting")
from scene import GaussianModel, Scene
from utils.gaussians import blend_gaussian_with_color, convert_gaussian_grayscale, cancel_feature_rest

def save_gaussians(input_folder:Path, gaussians:GaussianModel, iteration:str, suffix:str):
    save_path = input_folder / "point_cloud" / f"{iteration}_{suffix}" / "point_cloud.ply"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {suffix} Gaussians to {save_path}")
    gaussians.save_ply(save_path, skip_extra=True)

    deform_src = input_folder / "deform" / f"{iteration}"
    if deform_src.exists():
        deform_dst = input_folder / "deform" / f"./{iteration}_{suffix}"
        # soft link the deform folder
        if os.path.islink(str(deform_dst)):
            os.unlink(str(deform_dst))
        os.symlink(os.path.abspath(str(deform_src)), os.path.abspath(str(deform_dst)))


def main(args):
    assert args.input.exists(), f"{args.input} does not exist"
    point_cloud_folder = args.input / "point_cloud"/ "iteration_*"
    ply_paths = glob(str(point_cloud_folder / "point_cloud.ply"))
    ply_paths = [p for p in ply_paths if re.match("^iteration_[0-9]*$", p.split("/")[-2]) is not None]
    ply_paths = natsorted(ply_paths)
    ply_path = ply_paths[-1]
    iteration = ply_path.split("/")[-2]
    print("Loading point cloud from", ply_path)
    gaussians = GaussianModel(sh_degree=3, dim_extra=args.feat_dim)
    gaussians.load_ply(ply_path)

    # First save the gaussians to the rgb_only version
    save_gaussians(args.input, gaussians, iteration, "rgb")

    if args.point_id_path is None:
        print("No point id path is provided. Now will blend gaussians with PCA and KMeans colorization.")

        features_extra = gaussians.get_features_extra.detach().cpu().numpy() # (N, feat_dim)
        pca = PCA(n_components=3)
        pca.fit(features_extra)
        colors = pca.transform(features_extra) # (N, 3)
        colors = (colors - np.min(colors, axis=0)) / (np.max(colors, axis=0) - np.min(colors, axis=0)) # Normalize to [0, 1]
        colors = np.clip(colors, 0, 1)
        convert_gaussian_grayscale(gaussians)
        cancel_feature_rest(gaussians)
        blend_gaussian_with_color(gaussians, colors, alpha=0.75)
        save_gaussians(args.input, gaussians, iteration, "featpca")

        gaussians = GaussianModel(sh_degree=3, dim_extra=args.feat_dim)
        gaussians.load_ply(ply_path)
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(features_extra)
        palette = distinctipy.get_colors(args.n_clusters, rng=0)
        palette = np.array(palette)
        colors = palette[kmeans.labels_]
        convert_gaussian_grayscale(gaussians)
        cancel_feature_rest(gaussians)
        blend_gaussian_with_color(gaussians, colors, alpha=0.75)
        save_gaussians(args.input, gaussians, iteration, f"featkmeans{args.n_clusters}")

        return

    # Load the point id
    point_labels = np.load(args.point_id_path)
    max_id = np.max(point_labels)
    palette = distinctipy.get_colors(max_id + 1, rng=0) # Also include id 0
    if -1 in point_labels:
        palette.append((0.6, 0.6, 0.6))
    palette = np.array(palette)
    colors = palette[point_labels]

    # # Visualize the point cloud using open3d
    # xyz = gaussians.get_xyz.detach().cpu().numpy()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    blend_gaussian_with_color(gaussians, colors, mask=point_labels != -1, alpha=0.75)

    # Save the blended gaussians to the output path
    # save_path = point_cloud_folder.parent / f"{iteration}_cluster_color" / "point_cloud.ply"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # print("Saving blended Gaussians to", save_path)
    # gaussians.save_ply(save_path, skip_extra=True)
    save_gaussians(args.input, gaussians, iteration, "cluster_color")

    # Load again and make it grayscale this time
    gaussians = GaussianModel(sh_degree=3, dim_extra=args.feat_dim)
    gaussians.load_ply(ply_path)
    convert_gaussian_grayscale(gaussians)
    save_gaussians(args.input, gaussians, iteration, "gray")
    # save_path = point_cloud_folder.parent / f"{iteration}_gray" / "point_cloud.ply"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # print("Saving grayscale Gaussians to", save_path)
    # gaussians.save_ply(save_path, skip_extra=True)

    
    cancel_feature_rest(gaussians)
    blend_gaussian_with_color(gaussians, colors, mask=point_labels != -1, alpha=0.5)
    save_gaussians(args.input, gaussians, iteration, "cluster_gray")
    # save_path = point_cloud_folder.parent / f"{iteration}_cluster_gray" / "point_cloud.ply"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # print("Saving blended grayscale Gaussians to", save_path)
    # gaussians.save_ply(save_path, skip_extra=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default=Path("./output/v10_kitchen_lseg"),
                        help="Path to the trained point cloud")
    parser.add_argument("--feat_dim", type=int, default=32,
                        help="The feat dim of the PCA model")
    
    parser.add_argument("--n_clusters", type=int, default=16,
                        help="the number of clusters used in KMeans clustering. ")
    
    parser.add_argument("--point_id_path", type=str, default=None, 
                        help="The path to a file store per-point class/cluster id. ")

    args = parser.parse_args()
    main(args)
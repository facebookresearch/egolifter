from collections import Counter
import copy
import os, sys, argparse
import pickle
from glob import glob
from pathlib import Path
import distinctipy
import imageio

from natsort import natsorted

sys.path.append("/home/qgu/local/gaussian-splatting")
from scene import GaussianModel

import argparse

def main(args: argparse.Namespace):
    # ply_folders = glob(str(args.input / "point_cloud"/ "iteration_*/"))
    
    if args.iteration is None:
        # Only match the folders that the * is a pure number
        ply_folders = glob(str(args.dir / "point_cloud"/ "iteration_[0-9]*"))
        ply_folders = natsorted(ply_folders)
        ply_folder = ply_folders[-1]
    else:
        ply_folders = glob(str(args.dir / "point_cloud"/ f"iteration_{args.iteration}"))
        assert len(ply_folders) == 1, "One and only one folder should be matched."
        ply_folder = ply_folders[0]

    ply_path = ply_folder + "/point_cloud.ply"
    
    gaussians = GaussianModel(sh_degree=3, dim_extra=args.feat_dim)
    gaussians.load_ply(ply_path)

    # Save the gaussians with only RGB channels
    ply_save_folder = ply_folder + "_rgb"
    os.makedirs(ply_save_folder, exist_ok=True)
    ply_save_path = ply_save_folder + "/point_cloud.ply"
    gaussians.save_ply(ply_save_path, skip_extra=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=Path, required=True,
                        help="Path to the result folder, the output of the GS training.")
    parser.add_argument("--feat_dim", type=int, default=32,
                        help="The dimension of the feature vector.")
    
    parser.add_argument("--iteration", type=str, default=None, 
                        help="Iteration to load. If None, load the last iteration.")

    args = parser.parse_args()
    
    main(args)
from glob import glob
import os, sys
import re

from natsort import natsorted
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from pathlib import Path

import tyro

import open3d as o3d
from plyfile import PlyData, PlyElement


def get_last_ply_path(input_path: Path, return_all: bool = False):
    input_path = Path(input_path)
    ply_paths = glob(str(input_path / "point_cloud"/ "iteration_*" / "point_cloud.ply"))
    ply_paths = [p for p in ply_paths if re.match("^iteration_[0-9]*$", p.split("/")[-2]) is not None]
    ply_paths = natsorted(ply_paths)
    ply_path = ply_paths[-1]

    if return_all:
        return ply_paths
    else:
        return ply_path

@dataclass
class VisGaussianPcd:
    input_folder: Path = Path("output/adt_v2/Apartment_release_multiskeleton_party_seq121/unc_2d_unet_baseline_contr16_thresh0.5")

    def main(self):
        print(self.input_folder)
        path = get_last_ply_path(self.input_folder)
        
        plydata = PlyData.read(path)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # Load SH coefficients
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        # Load extra features
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_extra_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==self.dim_extra, "Expected {} extra features, found {}".format(self.dim_extra, len(extra_f_names))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        print(xyz.shape, features_dc.shape, features_extra.shape)
        
    
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(VisGaussianPcd).main()
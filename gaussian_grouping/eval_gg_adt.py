'''
This script needs to be run in the gaussian-grouping environment. 
'''

import colorsys
import os
from pprint import pprint
import numpy as np
from tqdm import tqdm
import tyro
from natsort import natsorted
from dataclasses import dataclass
import torch
import argparse

os.chdir("/rvl-home/guqiao/src/gaussian-grouping")

from scene import GaussianModel as gg_GaussianModel
from scene import Scene as gg_Scene
from gaussian_renderer import render as gg_render


@dataclass
class EvalGaussianGrouping:
    ckpt_root: str = "/rvl-home/guqiao/src/gaussian-grouping/output/adt"
    data_root: str = "/rvl-home/guqiao/ldata/adt_processed"
    scene_name: str = "Apartment_release_multiskeleton_party_seq121"
    
    load_iter: int = -1
    
    @ torch.no_grad()
    def main(self) -> None:
        ckpt_folder = os.path.join(self.ckpt_root, self.scene_name)
        data_folder = os.path.join(self.data_root, self.scene_name)
        ckpt_args_path = os.path.join(ckpt_folder, "cfg_args")
        
        # Load the dataset arguments and overwrite the model path and source path
        dataset_args = argparse.Namespace(**parse_namespace(ckpt_args_path))
        dataset_args.model_path = ckpt_folder
        dataset_args.source_path = data_folder
        
        # Construct the pipeline arguments
        pipeline_args = argparse.Namespace(
            convert_SHs_python = False,
            compute_cov3D_python = False,
            debug = False,
        )
        
        pprint(vars(dataset_args))
        
        gaussians = gg_GaussianModel(dataset_args.sh_degree)
        scene = gg_Scene(dataset_args, gaussians, load_iteration=self.load_iter, shuffle=False)

        num_classes = dataset_args.num_classes
        
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(
            ckpt_folder, "point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth"))
        )
        
        bg_color = [1,1,1] if dataset_args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        

        # Try render with the gg codebase
        views = scene.getTestCameras()
        
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            results = gg_render(view, gaussians, pipeline_args, background)
            rendering = results["render"]
            rendering_obj = results["render_object"]
            
            logits = classifier(rendering_obj)
            pred_obj = torch.argmax(logits,dim=0)
            
            print(results.keys())
            break


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def parse_namespace(file_path):
    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Remove the 'Namespace(' prefix and the closing ')'
    content = content.replace('Namespace(', '').rstrip(')')
    
    # Split the content into key-value pairs
    pairs = content.split(', ')
    
    # Parse the key-value pairs and store them in a dictionary
    parsed_data = {}
    for pair in pairs:
        key, value = pair.split('=')
        # Attempt to evaluate the value as Python literal if possible
        try:
            parsed_data[key] = eval(value)
        except:
            parsed_data[key] = value
    
    return parsed_data



if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(EvalGaussianGrouping).main()
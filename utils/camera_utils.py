#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
from PIL import Image
import json

WARNED = False

def cam_pose_to_GS_Rt(transform_matrix):
    '''
    Convert a camera pose to R and T useable by the 3DGS rasterizer. 
    '''
    transform_matrix = np.linalg.inv(transform_matrix)
    R = transform_matrix[:3,:3]
    R = R.T
    t = transform_matrix[:3,3]
    return R, t

def GS_Rt_to_cam_pose(R, t):
    '''
    Convert the R and t to a camera pose. 
    '''
    transform_matrix = np.eye(4)
    transform_matrix[:3,:3] = R.T
    transform_matrix[:3,3] = t
    transform_matrix = np.linalg.inv(transform_matrix)
    return transform_matrix

def getResolution(resolution, image: Image.Image | np.ndarray, resolution_scale: int) -> tuple[int, int]:
    if isinstance(image, np.ndarray):
        orig_w, orig_h = image.shape[1], image.shape[0]
    else:
        orig_w, orig_h = image.size

    if resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * resolution)), round(orig_h/(resolution_scale * resolution))
    else:  # should be a type that converts to float
        if resolution == -1:
            if orig_w > 1600:
                # global WARNED
                # if not WARNED:
                #     print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                #         "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                #     WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    return resolution


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )
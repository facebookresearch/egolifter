

from pathlib import Path
import sys, os

import numpy as np

from scene.cameras import Camera
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from .base import SceneInfo, SceneType, getNerfppNorm, storePly, fetchPly

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_list = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path_full = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path_full).split(".")[0]

        cam = Camera(
            colmap_id=uid,
            uid=uid,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image_width=width,
            image_height=height,
            image_name=image_name,
            image_path=image_path_full,
            scale=1.0,
            camera_name='rgb',
            scene_folder=str(Path(images_folder).parent),
        )
        
        cam_list.append(cam)
    sys.stdout.write('\n')
    return cam_list


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cameras_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cameras = sorted(cameras_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cameras = [c for idx, c in enumerate(cameras) if idx % llffhold != 0]
        print("Using the same set of cameras for validation and test")
        test_cameras = [c for idx, c in enumerate(cameras) if idx % llffhold == 0]
        valid_cameras = [c for idx, c in enumerate(cameras) if idx % llffhold == 0]
    else:
        train_cameras = cameras
        test_cameras = []
        valid_cameras = []

    nerf_normalization = getNerfppNorm(train_cameras)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cameras,
                           valid_cameras=valid_cameras,
                           test_cameras=test_cameras,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           scene_type=SceneType.COLMAP,
                           )
    return scene_info
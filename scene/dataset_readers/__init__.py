import os

from .colmap import readColmapSceneInfo
from .nerf_synthetic import readNerfSyntheticInfo
from .aria import readAriaSceneInfo
from .replica import readReplicaInfo
from .replica_semantic import readReplicaSemanticInfo
from .nerfies import readNerfiesInfo

from .aggregate import aggregate_scene_infos
from .base import BasicPointCloud, SceneType, storePly

def get_scene_info(cfg_scene):
    assert os.path.exists(cfg_scene.source_path), f"Source path {cfg_scene.source_path} does not exist!"

    if os.path.exists(os.path.join(cfg_scene.source_path, "sparse")):
        scene_info = readColmapSceneInfo(cfg_scene.source_path, cfg_scene.images, cfg_scene.eval)
        
    elif os.path.exists(os.path.join(cfg_scene.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = readNerfSyntheticInfo(cfg_scene.source_path, cfg_scene.white_background, cfg_scene.eval)

    elif os.path.exists(os.path.join(cfg_scene.source_path, "global_points.csv.gz")):
        print("Found global_points.csv.gz file, assuming Aria data set!")
        # scene_info = readAriaSceneInfo(cfg_scene.source_path, cfg_scene.camera_name, cfg_scene.stride, cfg_scene.scene_name)
        scene_info = readAriaSceneInfo(cfg_scene)
        
    elif os.path.exists(os.path.join(cfg_scene.source_path, "transforms.json")):
        print("Found transforms.json file, assuming Aria data set!")
        # scene_info = readAriaSceneInfo(cfg_scene.source_path, cfg_scene.camera_name, cfg_scene.stride, cfg_scene.scene_name)
        scene_info = readAriaSceneInfo(cfg_scene)

    elif os.path.exists(os.path.join(cfg_scene.source_path, "traj_w_c.txt")):
        print("Found traj_w_c.txt file. Assuming Replica Semantic dataset!")
        scene_info = readReplicaSemanticInfo(cfg_scene.source_path, image_stride=cfg_scene.stride, pcd_stride=cfg_scene.pcd_stride)
    
    elif os.path.exists(os.path.join(cfg_scene.source_path, "traj.txt")):
        print("Found traj.txt, assuming Replica data set!")
        scene_info = readReplicaInfo(cfg_scene.source_path, image_stride=cfg_scene.stride, pcd_stride=cfg_scene.pcd_stride)

    elif os.path.exists(os.path.join(cfg_scene.source_path, "dataset.json")):
        print("Found dataset.json file, assuming Nerfies data set!")
        scene_info = readNerfiesInfo(cfg_scene.source_path, cfg_scene.eval)
        
    else:
        assert False, "Could not recognize scene type!"
    
    return scene_info
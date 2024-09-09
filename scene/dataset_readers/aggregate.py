import tempfile

import numpy as np
from scene.gaussian_model import BasicPointCloud
from scene.cameras import Camera

from .base import SceneInfo, SceneType, getNerfppNorm, storePly, fetchPly


def aggregate_scene_infos(scene_infos): 
    """
    Aggregate multiple scene infos into one 
    """
    points_agg = []
    colors_agg = []
    normals_agg= []
    train_cameras = []
    valid_cameras = []
    test_cameras = []
    valid_novel_cameras = []
    scene_type = None
    valid_mask_by_name = None 
    vignette_by_name = None
    
    for scene_info in scene_infos:    
        points_agg.append(scene_info.point_cloud.points) 
        colors_agg.append(scene_info.point_cloud.colors)
        normals_agg.append(scene_info.point_cloud.normals)

        train_cameras += scene_info.train_cameras
        valid_cameras += scene_info.valid_cameras
        test_cameras += scene_info.test_cameras
        valid_novel_cameras += scene_info.valid_novel_cameras

        if scene_type is None: 
            scene_type = scene_info.scene_type 
        else: 
            assert scene_type == scene_info.scene_type, \
                "aggregated scene infos need to have the same scene type"

        # will share mask and vignette image for all frames
        if valid_mask_by_name is None: 
            valid_mask_by_name = scene_info.valid_mask_by_name

        if vignette_by_name is None:
            vignette_by_name = scene_info.vignette_by_name 
            
        # TODO: aggregate the segmentation meta information
        if scene_info.seg_static_ids is not None:
            raise NotImplementedError("Aggregated Segmentation Info is not supported, yet. ")
            
    # Limit the number of validation images for a single scene to 5000
    if len(valid_cameras) > 5000:
        valid_cameras = [valid_cameras[i] for i in np.linspace(0, len(valid_cameras)-1, 5000, dtype=int)]

    points_agg  = np.concatenate(points_agg, axis=0)
    colors_agg  = np.concatenate(colors_agg, axis=0)
    normals_agg = np.concatenate(normals_agg,axis=0)
    
    pcd = BasicPointCloud(points=points_agg, colors=colors_agg, normals=normals_agg)

    ply_path = tempfile.NamedTemporaryFile(suffix=".ply", delete=False).name
    storePly(ply_path, points_agg, colors_agg, normals_agg)

    nerf_normalization = getNerfppNorm(train_cameras)

    scene_info_agg = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cameras,
        valid_cameras=valid_cameras,
        test_cameras=test_cameras,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        scene_type=scene_type,
        valid_novel_cameras=valid_novel_cameras,
        valid_mask_by_name=valid_mask_by_name,
        vignette_by_name=vignette_by_name,
        seg_static_ids=None,
        seg_dynamic_ids=None,
        query_2dseg=None,
    )

    return scene_info_agg
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from omegaconf import DictConfig, OmegaConf
import torch

from scene import Scene
from model import get_model
from utils.system_utils import get_latest_ckpt

def load_from_model_path(
    model_path, 
    source_path = None, 
    data_root: str = None,
    ckpt_subpath: str = None,
    simple_scene: bool = False
):
    assert model_path is not None, "model_path must be specified"
    assert os.path.exists(model_path), f"model_path {model_path} does not exist"
    
    path_cfg = os.path.join(model_path, "config.yaml")
    cfg = OmegaConf.load(open(path_cfg, "r"))
    
    # TODO: better way to handle newly-added config
    cfg.scene.no_novel = False
    cfg.scene.all_seen_train = False
    
    assert source_path is None or data_root is None, "Only one of source_path and data_root can be specified"
    
    if source_path is not None:
        cfg.scene.source_path = source_path
        print(f"Overriding source_path to {source_path}")
    if data_root is not None:
        cfg.scene.data_root = data_root
        cfg.scene.source_path = f"{cfg.scene.data_root}/{cfg.scene.scene_name}"
    
    try:
        scene = Scene(cfg, simple=simple_scene)
    except Exception as e:
        print(f"Failed to initialize scene from {cfg.scene.source_path}")
        print(e.with_traceback(None))
        print("Try proceeding with scene set to None")
        scene = None

    model = get_model(cfg, scene)
    
    if ckpt_subpath is None:
        ckpt = get_latest_ckpt(model_path)
    else:
        ckpt_path = os.path.join(model_path, ckpt_subpath)
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
    
    model.init_gaussians_size_from_state_dict(ckpt['state_dict'])
    model.load_state_dict(ckpt['state_dict'])
    model.to("cuda").eval()

    return model, scene, cfg
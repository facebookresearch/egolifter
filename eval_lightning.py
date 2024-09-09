import os
from glob import glob
from typing import Any

from natsort import natsorted

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import torch
import torchvision
import lightning as L
from tqdm import tqdm

from utils.routines import load_from_model_path


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg_eval : DictConfig) -> None:
    model, scene, cfg = load_from_model_path(cfg_eval.model_path, source_path=cfg_eval.source_path)
    save_folder = cfg.scene.model_path

    if cfg_eval.eval_on_gg:
        from model.gaussian_grouping import GaussianGrouping
        assert cfg_eval.gg_ckpt_folder is not None, "gg_ckpt_folder must be specified"
        print(f"Loading GaussainGrouping model from {cfg_eval.gg_ckpt_folder}")
        cfg.model.name = "gaussian_grouping"
        cfg.model.gg_ckpt_folder = cfg_eval.gg_ckpt_folder
        save_folder = cfg_eval.gg_ckpt_folder
        model = GaussianGrouping(cfg, scene)
    
    trainer = L.Trainer(
        devices=cfg.gpus, 
    )
    
    for subset in ["test", "valid"]:
        loader = scene.get_data_loader(subset, shuffle=False)
        if len(loader) > 0:
            trainer.test(
                model=model,
                dataloaders=loader,
            )
            df = pd.DataFrame(model.test_logs)
            df.to_csv(os.path.join(save_folder, f"eval_logs_{subset}.csv"), index=False)



if __name__ == "__main__":
    main()
import os
from pathlib import Path
import cv2 # This import is needed for CCDB cluster

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

import wandb

from model import get_model
from scene import Scene
from callback.checkpoint import ModelCheckpoint

from utils.eval_2dseg import eval_query_2dseg

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg : DictConfig) -> None:
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    # print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.scene.model_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.scene.model_path, "wandb"), exist_ok=True)
    
    
    os.makedirs(cfg.wandb.save_dir, exist_ok=True)
    logger = WandbLogger(
        project=cfg.wandb.project, 
        entity=cfg.wandb.entity,
        name=cfg.exp_name,
        save_dir=cfg.wandb.save_dir,
    )
    
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(cfg, os.path.join(cfg.scene.model_path, "config.yaml"), resolve=True)
    
    L.seed_everything(cfg.seed)
    
    scene = Scene(cfg)
    scene.save_camera_json()
    
    model = get_model(cfg, scene)

    model.init_or_load_gaussians(
        scene.scene_info.point_cloud,
        scene.scene_info.nerf_normalization["radius"],
        cfg.scene.model_path,
        load_iteration = None,
    )

    train_loader = scene.get_data_loader("train", shuffle=True, num_workers=cfg.scene.num_workers)
    valid_loader = scene.get_data_loader("valid", shuffle=False, num_workers=cfg.scene.num_workers)
    valid_novel_loader = scene.get_data_loader("valid_novel", shuffle=False, num_workers=cfg.scene.num_workers)
    
    # TODO: Resume from an existing checkpoint, if needed (currently it won't work)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.scene.model_path,
        filename="chkpnt{step}",
        save_top_k=-1,
        verbose=True,
        monitor=None,
        every_n_train_steps = cfg.opt.ckpt_every_n_steps,
    )

    trainer = L.Trainer(
        max_steps=cfg.opt.iterations,
        logger=logger,
        check_val_every_n_epoch=None,
        val_check_interval = cfg.opt.val_every_n_steps, # validation after every 5000 steps
        callbacks=[checkpoint_callback],
        devices=cfg.gpus, 
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=[valid_loader, valid_novel_loader],
    )
    
    #### Evaluation ####
    # train set can be much larger than validation set, so we evaluate on valid set first
    if not cfg.skip_test:
        for subset in ["test", "valid", "valid_novel", "train"]:
            loader = scene.get_data_loader(subset, shuffle=False)
            if len(loader) > 0:
                trainer.test(
                    model=model,
                    dataloaders=loader,
                )
                df = pd.DataFrame(model.test_logs)
                df.to_csv(os.path.join(cfg.scene.model_path, f"eval_logs_{subset}.csv"), index=False)
                
    #### Evaluate the 2D segmentation ####
    if cfg.lift.use_contr:
        if scene.scene_info.query_2dseg is None:
            print("No 2D segmentation query found in the scene info. Skipping 2D segmentation evaluation.")
        else:
            # Copy the to a new model to avoid a weird memory illegal access error
            model_eval = get_model(cfg, scene)
            model_eval.init_gaussians_size_from_state_dict(model.state_dict())
            model_eval.load_state_dict(model.state_dict())
            model_eval = model_eval.eval().cuda()
            
            for subset in ["test", "valid", "valid_novel", "train"]:
                print(f"Evaluating subset: {subset} ...")
                dataloader = scene.get_data_loader(subset, shuffle=False, num_workers=0, limit=200)
                threshold_mode = "fixed"
                threshold_value = 0.6
                static_miou, dynamic_miou, df_eval_logs = eval_query_2dseg(
                    scene, dataloader, model_eval, threshold_mode, threshold_value)
                print(f"{subset}: static mIoU: {static_miou}, dynamic mIoU: {dynamic_miou}")
                
                wandb.log({
                    f"2dseg_static/{subset}_miou": static_miou,
                    f"2dseg_dynamic/{subset}_miou": dynamic_miou,
                })
                
                # Save the evaluation logs to the ckpt_folder
                save_path = Path(cfg.scene.model_path) / "2dseg_eval" / f"{subset}_logs_{threshold_mode}_{threshold_value}.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                df_eval_logs.to_csv(save_path, index=False)
                
        

if __name__ == "__main__":
    main()
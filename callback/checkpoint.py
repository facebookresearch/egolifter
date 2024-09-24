# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import lightning as L

class ModelCheckpoint(L.pytorch.callbacks.ModelCheckpoint):
    def _save_checkpoint(self, trainer: "L.Trainer", filepath: str) -> None:
        '''
        Here we customize the checkpointing behavior such that it also saves the Gaussians to a ply file.
        This is purely for backward compatibility, but it's not an optimal behavior. 
        '''
        super()._save_checkpoint(trainer, filepath)
        
        # Then save the ply file to another directory
        step = trainer.global_step
        save_path = os.path.join(self.dirpath, "point_cloud", f"iteration_{step}", "point_cloud.ply")
        trainer.model.save_point_cloud(save_path)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def get_model(cfg, scene):
    if cfg.model.name == "vanilla":
        from .vanilla import VanillaGaussian
        return VanillaGaussian(cfg, scene)
    elif cfg.model.name == "unc_2d_unet":
        from .unc_2d_unet import Unc2DUnet
        return Unc2DUnet(cfg, scene)
    elif cfg.model.name == "deform":
        from .deform import DeformGaussian
        return DeformGaussian(cfg, scene)
    elif cfg.model.name == "gaussian_grouping":
        from .gaussian_grouping import GaussianGrouping
        return GaussianGrouping(cfg, scene)
    else:
        raise Exception(f"ERROR: Model {cfg.model.name} not implemented. ")
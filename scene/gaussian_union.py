# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from .gaussian_model import GaussianModel


class GaussianUnion():
    '''
    A class that concatenates multiple Gaussian models into a single one.
    Only designed for the usage in the rasterization process. 
    '''
    def __init__(self, gaussian_models: list[GaussianModel]) -> None:
        self.gaussian_models = gaussian_models
        
        self.active_sh_degree = gaussian_models[0].active_sh_degree
        self.dim_extra = gaussian_models[0].dim_extra
        
        
    @property
    def get_scaling(self):
        return torch.cat([m.get_scaling for m in self.gaussian_models], dim=0)
    
    @property
    def get_rotation(self):
        return torch.cat([m.get_rotation for m in self.gaussian_models], dim=0)
    
    @property
    def get_xyz(self):
        return torch.cat([m.get_xyz for m in self.gaussian_models], dim=0)
    
    @property
    def get_features(self):
        return torch.cat([m.get_features for m in self.gaussian_models], dim=0)
    
    @property
    def get_features_dc(self):
        return torch.cat([m.get_features_dc for m in self.gaussian_models], dim=0)
    
    def get_rgb(self):
        return torch.cat([m.get_rgb() for m in self.gaussian_models], dim=0)
    
    @property
    def get_features_rest(self):
        return torch.cat([m.get_features_rest for m in self.gaussian_models], dim=0)
    
    @property
    def get_features_extra(self):
        return torch.cat([m.get_features_extra for m in self.gaussian_models], dim=0)
    
    @property
    def get_opacity(self):
        return torch.cat([m.get_opacity for m in self.gaussian_models], dim=0)
    
    def get_covariance(self, scaling_modifier = 1):
        return torch.cat([m.get_covariance(scaling_modifier) for m in self.gaussian_models], dim=0)
    
    def get_covariance_matrix(self, scaling_modifier = 1):
        # return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)[1]
        return torch.cat([m.get_covariance_matrix(scaling_modifier) for m in self.gaussian_models], dim=0)
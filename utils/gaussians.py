import torch
import torch.nn as nn
from torchvision.transforms.functional import adjust_saturation

import numpy as np
from sklearn.decomposition import PCA

from utils.sh_utils import RGB2SH, SH2RGB
from scene import GaussianModel


def convert_gaussian_feat_pca(gaussians: GaussianModel, alpha=0.75, normalize_max=1.0):
    assert gaussians.get_features_extra.shape[1] > 0, "No extra features to convert"
    features_extra = gaussians.get_features_extra.detach().cpu().numpy() # (N, feat_dim)
    pca = PCA(n_components=3)
    pca.fit(features_extra)
    colors = pca.transform(features_extra) # (N, 3)
    colors = (colors - np.min(colors, axis=0)) / (np.max(colors, axis=0) - np.min(colors, axis=0)) # Normalize to [0, 1]
    colors = np.clip(colors, 0, 1)
    # colors = colors * normalize_max
    convert_gaussian_grayscale(gaussians)
    cancel_feature_rest(gaussians)
    blend_gaussian_with_color(gaussians, colors, alpha=alpha)
    
def improve_saturation(gaussians: GaussianModel, saturation_factor=1.5):
    features_dc = gaussians.get_features_dc.detach() # (N, 1, 3)
    colors = SH2RGB(features_dc) # (N, 1, 3)
    colors = colors.permute(2, 0, 1) # (3, N, 1)
    colors = adjust_saturation(colors, saturation_factor)
    colors = colors.permute(1, 2, 0) # (N, 1, 3)
    features_dc = RGB2SH(colors) # (N, 1, 3)
    gaussians._features_dc = nn.Parameter(features_dc, requires_grad=True) # (N, 1, 3)


def convert_gaussian_grayscale(gaussians):
    features_dc = gaussians.get_features_dc.detach()
    features_rest = gaussians.get_features_rest.detach()

    features_dc[:] = torch.mean(features_dc, dim=-1, keepdim=True).repeat(1, 1, features_dc.shape[-1])
    features_rest[:] = torch.mean(features_rest, dim=-1, keepdim=True).repeat(1, 1, features_rest.shape[-1])

    # assert features_dc_gray.shape == gaussians.get_features_dc.shape
    # assert features_rest_gray.shape == gaussians.get_features_rest.shape

    # gaussians.get_feature_dc = nn.Parameter(features_dc_gray, requires_grad=True)
    # gaussians.get_feature_rest = nn.Parameter(features_rest_gray, requires_grad=True)

def blend_gaussian_with_color(gaussians, colors, mask=None, alpha=0.5):
    colors_sh = RGB2SH(colors)
    colors_sh = torch.from_numpy(colors_sh).to(gaussians.get_features_dc)
    features_dc = gaussians.get_features_dc.detach()
    colors_sh = colors_sh.unsqueeze(1)

    if mask is None:
        mask = torch.ones(features_dc.shape[0], dtype=torch.bool, device=features_dc.device)
    
    features_dc[mask] = features_dc[mask] * (1-alpha) + colors_sh[mask] * alpha

    # gaussians._feature_dc = nn.Parameter(features_dc, requires_grad=True)

def cancel_feature_rest(gaussians):
    features_rest = gaussians.get_features_rest.detach()
    features_rest[:] = 0.0

    # gaussians._feature_rest = nn.Parameter(features_rest, requires_grad=True)
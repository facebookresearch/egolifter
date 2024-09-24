# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import gzip
import os
import pickle
from typing import Optional
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn.functional as F
from scene.cameras import Camera

from scene.dataset_readers import SceneType

class ContrastManager():
    def __init__(
        self, 
        cfg: DictConfig, 
        example_cam: Camera, # A training camera 
        valid_mask_by_name: Optional[dict],
        scene_type: SceneType,
    ) -> None:
        self.cfg = cfg
        self.scene_type = scene_type
        self.in_use = cfg.lift.use_contr
        self.n_samples = cfg.lift.n_samples

        if not self.in_use:
            # Do not perform contrastive lifting
            return
        
        # Get the size of the segmentation images and use its size for feature rendering. 
        try:
            mask_image = get_mask_onelabel(
                example_cam.image_name,
                example_cam.scene_folder, 
                cfg.lift.det_folder_name,
                resize_to=None
            )
        except AssertionError as e:
            print(e)
            # Try to load the mask with the "train" prefix (For D-NeRF)
            mask_image = get_mask_onelabel(
                "train/" + example_cam.image_name,
                example_cam.scene_folder, 
                cfg.lift.det_folder_name,
                resize_to=None
            )

        seg_mask_size = mask_image.shape[:2] # (H, W)

        # Only sample pixels falling on the valid mask
        if valid_mask_by_name:
            # Use the example_cam to get the valid mask
            # TODO: handle multiple valid masks when we decide to use them for training. 
            name = example_cam.valid_mask_subpath
            valid_mask = valid_mask_by_name[name][1.0] # (1, H, W)
            valid_mask = valid_mask.squeeze(0) # (H, W)
            valid_mask = F.interpolate(
                valid_mask.unsqueeze(0).unsqueeze(0), 
                size=seg_mask_size, mode='bilinear'
            ).squeeze(0).squeeze(0) # (H, W)
            valid_indices_all = (valid_mask > 0.75).nonzero(as_tuple=False) # (N, 2), handle the non binary mask on the border
            valid_indices_all_flatten = valid_indices_all[:, 0] * valid_mask.shape[1] + valid_indices_all[:, 1] # (N,)
        else:
            # All pixels on the images are valid
            valid_indices_all_flatten = torch.arange(
                seg_mask_size[0] * seg_mask_size[1],
            ).cuda()
    
        self.seg_mask_size = seg_mask_size
        self.valid_indices_all_flatten = valid_indices_all_flatten

    def get_feat_cam(self, viewpoint_cam: Camera) -> Camera:
        # Set the rendering size to match the mask_image
        viewpoint_cam_mask = viewpoint_cam.copy()
        viewpoint_cam_mask.image_height = self.seg_mask_size[0]
        viewpoint_cam_mask.image_width = self.seg_mask_size[1]
        return viewpoint_cam_mask
    
    def get_mask(self, viewpoint_cam, subpath = None):
        assert self.cfg.lift.det_folder_name is not None, "Please specify the detection folder name"

        image_name = viewpoint_cam.image_name
        if subpath is not None:
            image_name = os.path.join(subpath, image_name)
            image_path = os.path.join(
                viewpoint_cam.scene_folder, 
                self.cfg.lift.det_folder_name, 
                image_name + ".pkl.gz"
            )
            if not os.path.exists(image_path):
                image_name = viewpoint_cam.image_name

        if self.cfg.lift.contr_multilabel:
            mask_image = get_mask_multilabel(
                image_name, 
                viewpoint_cam.scene_folder, 
                self.cfg.lift.det_folder_name,
                resize_to=self.seg_mask_size,
                rotate_aria = self.scene_type == SceneType.ARIA,
            ) # (H, W, L)
        else:
            mask_image = get_mask_onelabel(
                image_name, 
                viewpoint_cam.scene_folder, 
                self.cfg.lift.det_folder_name,
                resize_to=self.seg_mask_size,
                rotate_aria = self.scene_type == SceneType.ARIA,
            ) # (H, W)

        return mask_image

    def compute_loss(
            self, 
            gt_mask_image: torch.Tensor, 
            render_features: torch.Tensor, 
            temperature: float,
            weight_image: torch.Tensor = None,
        ) -> torch.Tensor:
        if not self.in_use:
            # Do not perform contrastive lifting
            return torch.tensor(0.0)
        
        if gt_mask_image.numel() <= 1:
            # Skip contrastive loss if the mask is empty
            print("Empty mask, skipping contrastive loss")
            return torch.tensor(0.0)
        
        assert render_features.shape[:2] == gt_mask_image.shape[:2], f"render_features.shape {render_features.shape} != gt_mask_image.shape {gt_mask_image.shape}"

        render_features_flatten = render_features.reshape(-1, render_features.shape[-1]) # (H * W, D)
        sample_idx = self.valid_indices_all_flatten[torch.randperm(self.valid_indices_all_flatten.shape[0])[:self.n_samples]]

        # Handle multi-label differencely 
        if self.cfg.lift.contr_multilabel:
            mask_flatten = gt_mask_image.reshape(-1, gt_mask_image.shape[-1]) # (H * W, L)
            sample_idx = sample_idx[mask_flatten.amax(-1)[sample_idx] > 0] # ensure using foreground pixels
        else:
            mask_flatten = gt_mask_image.reshape(-1) # (H * W)
            sample_idx = sample_idx[mask_flatten[sample_idx] >= 0] # ensure using foreground pixels

        # Get the weight for each sampled pixel
        sample_weights = None
        if weight_image is not None:
            weight_flatten = weight_image.reshape(-1) # (H * W)
            assert weight_flatten.shape[0] == mask_flatten.shape[0], f"weight_flatten.shape {weight_flatten.shape} != mask_flatten.shape {mask_flatten.shape}"
            sample_weights = weight_flatten[sample_idx]
            # Apply the thresholding weight if told so
            if weight_image is not None and self.cfg.model.contr_weight_mode == "thresh":
                sample_idx = sample_idx[sample_weights > self.cfg.model.contr_weight_thresh]
                sample_weights = sample_weights[sample_weights > self.cfg.model.contr_weight_thresh]
            
        # If there are too few pixels sampled, skip contrastive loss for now
        if len(sample_idx) < 2 ** 3:
            return torch.tensor(0.0)

        sampled_mask_flatten = mask_flatten[sample_idx] # (N, ) or (N, L)

        # Contrastive loss on the rendered features
        loss_contr = contrastive_loss(
            render_features_flatten[sample_idx],
            sampled_mask_flatten,
            temperature = temperature,
            weight = sample_weights,
            sum_in_log = not self.cfg.lift.sum_out_log,
            sim_exp = self.cfg.lift.sim_exp,
            weighting_mode = self.cfg.model.contr_weight_mode,
        )

        return loss_contr

    

def get_mask_onelabel(
        image_name, 
        source_path, 
        det_folder_name, 
        resize_to = None,
        rotate_aria = False,
    ):
    if image_name.endswith(".png") or image_name.endswith(".jpg"):
        image_name = image_name[:-4]
    det_path = os.path.join(source_path, det_folder_name, image_name + ".pkl.gz")
    assert os.path.exists(det_path), f"Could not find detection file at {det_path}"
    # det_path = glob.glob(os.path.join(source_path, det_folder_name, "**/*.pkl.gz"))[0]
    # assert os.path.exists(det_path), f"Could not find detection file at {det_path}"

    with gzip.open(det_path, 'rb') as f:
        det_result = pickle.load(f)

    # Convert the mask from a list of masks to a single image with per-pixel integer labels
    # mask_image = torch.zeros(det_result['mask'][0].shape, dtype=torch.int32)
    if len(det_result['mask']) == 0:
        return torch.zeros(1, dtype=torch.int32)

    mask_image = np.ones(det_result['mask'][0].shape, dtype=np.int32) * -1
    for mask_idx, mask in enumerate(det_result['mask']):
        mask_image[mask] = mask_idx

    if rotate_aria:
        # IMPORTANT: GSA results are upright, and thus we need to rotate them back to the Aria orientation
        mask_image = np.rot90(mask_image, k=1, axes=(0, 1)).copy()
    mask_image = torch.from_numpy(mask_image) # (H, W)

    if resize_to is not None and mask_image.shape != resize_to:
        # Interpolate the mask_image to resize_to
        mask_image = F.interpolate(
            mask_image.unsqueeze(0).unsqueeze(0).float(), 
            size=resize_to, mode='nearest'
        ).squeeze(0).squeeze(0).long() # (H, W)

    return mask_image


def get_mask_multilabel(
        image_name, 
        source_path, 
        det_folder_name, 
        resize_to = None,
        rotate_aria = False,
    ):
    if image_name.endswith(".png") or image_name.endswith(".jpg"):
        image_name = image_name[:-4]
    det_path = os.path.join(source_path, det_folder_name, image_name + ".pkl.gz")
    assert os.path.exists(det_path), f"Could not find detection file at {det_path}"

    with gzip.open(det_path, 'rb') as f:
        det_result = pickle.load(f)

    mask_image = np.stack(det_result['mask'], axis=-1) # (H, W, L)
    mask_image = mask_image.astype(bool) # (H, W, L)

    if rotate_aria:
        # IMPORTANT: GSA results are upright, and thus we need to rotate them back to the Aria orientation
        mask_image = np.rot90(mask_image, k=1, axes=(0, 1)).copy()
    mask_image = torch.from_numpy(mask_image) # (H, W, L)

    if resize_to is not None and mask_image.shape[:2] != resize_to:
        # # Interpolate the mask_image to resize_to
        mask_image = F.interpolate(
            mask_image.permute(2, 0, 1).unsqueeze(0).float(),
            size=resize_to, mode='bilinear'
        ).squeeze(0).permute(1, 2, 0) # (H, W, L)
        mask_image = mask_image > 0.5 # (H, W, L)

    return mask_image
    

def contrastive_loss(
        features: torch.Tensor, 
        instance_labels: torch.Tensor, 
        temperature: float, 
        weight: Optional[torch.Tensor] = None,
        sum_in_log: bool = True, 
        sim_exp: float = 1.0,
        weighting_mode: str = "none",
    ):
    '''
    Args:
        features: (N, D), features of the sampled pixels
        instance_labels: (N, ) or (N, L), integer labels or multi-hot bool labels
        temperature: temperature for the softmax
        weight: (N, ), weight for each sample in the final loss
        sum_in_log: whether to sum in log space or not
        sim_exp: exponent for the similarity
        weighting_mode: "on_sim" or "on_prob"
    '''
    assert features.shape[0] == instance_labels.shape[0], f"{features.shape}, {instance_labels.shape} does not match. "
    if weight is not None:
        assert features.shape[0] == weight.shape[0], f"{features.shape}, {weight.shape} does not match. "
    
    bsize = features.size(0) # N
    if instance_labels.dim() == 1: # (N, ), integer labels
        sim_masks = instance_labels.view(-1, 1).repeat(1, bsize).eq_(instance_labels.clone()) # (N, N)
        sim_masks = sim_masks.fill_diagonal_(0, wrap=False) # (N, N)
    elif instance_labels.dim() == 2: # (N, L), multi-hot labels, in bool type
        inter = instance_labels.unsqueeze(1) & instance_labels.unsqueeze(0) # (N, N, L) 
        union = instance_labels.unsqueeze(1) | instance_labels.unsqueeze(0) # (N, N, L)
        sim_masks = inter.float().sum(dim=-1) / union.float().sum(dim=-1) # (N, N)
        sim_masks = sim_masks.fill_diagonal_(0, wrap=False) # (N, N)
        sim_masks = sim_masks ** sim_exp # (N, N)
    else:
        raise Exception(f"instance_labels.dim() {instance_labels.dim()} is not supported")

    # compute similarity matrix based on Euclidean distance
    distance_sq = torch.pow(features.unsqueeze(1) - features.unsqueeze(0), 2).sum(dim=-1) # (N, N)

    # temperature = 1 for positive pairs and temperature (100) for negative pairs
    temperature = torch.ones_like(distance_sq) * temperature # (N, N)
    temperature = torch.where(sim_masks==1, temperature, torch.ones_like(temperature)) # (N, N)

    # Process and apply the weights
    if weight is not None:
        weight_matrix = weight.unsqueeze(1) * weight.unsqueeze(0) # (N, N)

    similarity_kernel = torch.exp(-distance_sq/temperature) # (N, N)
    if weight is not None and weighting_mode == "on_sim":
        similarity_kernel = similarity_kernel * weight_matrix # (N, N)
        
    prob_before_norm = torch.exp(similarity_kernel) # (N, N)
    if weight is not None and weighting_mode == "on_prob":
        prob_before_norm = prob_before_norm * weight_matrix # (N, N)

    if sum_in_log: 
        # First sum over positive pairs and then log - better in handling noises. 
        Z = prob_before_norm.sum(dim=-1) # (N,), denom
        p = torch.mul(prob_before_norm, sim_masks).sum(dim=-1) # (N,), numer
        prob = torch.div(p, Z) # (N,)
        prob_masked = torch.masked_select(prob, prob.ne(0)) # (N,)
        loss = - prob_masked.log().sum()/bsize # (1,)
    else:
        # First take the log and then sum over positive pairs - forcing more precise matching. 
        Z = prob_before_norm.sum(dim=-1, keepdim=True) # (N, N), denom
        prob = torch.div(prob_before_norm, Z) # (N, N)
        log_prob = torch.log(prob) # (N, N)

        # Get only the positive pairs (with similarity larger than 0)
        weighted_log_prob = torch.mul(log_prob, sim_masks) # (N, N)
        # Normalized by the number of positive pairs for each anchor
        weighted_log_prob = weighted_log_prob / (sim_masks.ne(0).sum(-1, keepdim=True) + 1e-6) # (N, N)
        # Sum over positive pairs
        log_prob_masked = torch.masked_select(weighted_log_prob, weighted_log_prob.ne(0)) # (N,)
        # Sum over anchors and normalized by the batch size
        loss = - log_prob_masked.sum()/bsize # (1, )

    return loss

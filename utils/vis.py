# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import cv2
import numpy as np

import torch

from sklearn.decomposition import PCA

def resize_image(image: np.ndarray, max_length: int):
    # image = image.clip(0.0, 1.0)
    h, w = image.shape[:2]
    if max([h, w]) != max_length:
        if h > w:
            new_h, new_w = max_length, int(max_length * w / h)
        else:
            new_h, new_w = int(max_length * h / w), max_length
        image = cv2.resize(image, (new_w, new_h))
    return image

def feat_image_to_pca_vis(
    feat_image: torch.Tensor,
    channel_first: bool = True,
):
    if channel_first: # input is [D, H, W]
        shape = feat_image.shape
        feat_image = feat_image.permute(*range(1, len(shape)), 0) # [H, W, D]
        
    shape = feat_image.shape # [H, W, D]
        
    feat_image = feat_image.contiguous().cpu().numpy() # [H, W, D]
    feat_image = feat_image.reshape(-1, feat_image.shape[-1]) # [H * W, D]
    pca_image = PCA(3).fit_transform(feat_image) # [H * W, 3]

    pca_min = pca_image.min(axis=0)
    pca_max = pca_image.max(axis=0)
    pca_image = (pca_image - pca_min) / (pca_max - pca_min + 1e-6)
    pca_image = pca_image.reshape(*shape[:-1], 3) # [H, W, 3]

    return pca_image
    

def concate_images_vis(
    images: list[np.ndarray],
):
    '''
    Concate images into one image for visualization.

    Args:
        images: list of images, each image is a numpy array with shape [H, W, 3], [H, W, 1] or [H, W]
    '''
    if not images:
        raise ValueError("The input list is empty.")
    
    # get the maximum size of all images
    max_hw = np.max([image.shape[:2] for image in images], axis=0)
    
    processed_images = []
    for image in images:
        if not isinstance(image, np.ndarray):
            raise TypeError("All images should be numpy arrays.")
        if image.ndim < 2 or image.ndim > 3:
            raise ValueError("All images should have 2 or 3 dimensions.")
        
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        resize_scale = np.min(max_hw / image.shape[:2])
        image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        
        # pad the image to the maximum size
        if image.shape[0] < max_hw[0]:
            left_pad = (max_hw[0] - image.shape[0]) // 2
            right_pad = max_hw[0] - image.shape[0] - left_pad
            image = np.pad(image, ((left_pad, right_pad), (0, 0), (0, 0)), mode='constant')
        if image.shape[1] < max_hw[1]:
            top_pad = (max_hw[1] - image.shape[1]) // 2
            bottom_pad = max_hw[1] - image.shape[1] - top_pad
            image = np.pad(image, ((0, 0), (top_pad, bottom_pad), (0, 0)), mode='constant')
            
        processed_images.append(image)
        
    if len(processed_images) <= 3:
        concate_image = np.concatenate(processed_images, axis=1)
    elif len(processed_images) <= 6:
        first_row = np.concatenate(processed_images[:3], axis=1)
        second_row = np.concatenate(processed_images[3:], axis=1)
        second_row_padded = np.pad(second_row, ((0, 0), (0, first_row.shape[1] - second_row.shape[1]), (0, 0)), mode='constant')
        concate_image = np.concatenate([first_row, second_row_padded], axis=0)
    elif len(processed_images) <= 9:
        first_row = np.concatenate(processed_images[:3], axis=1)
        second_row = np.concatenate(processed_images[3:6], axis=1)
        third_row = np.concatenate(processed_images[6:], axis=1)
        third_row_padded = np.pad(third_row, ((0, 0), (0, first_row.shape[1] - third_row.shape[1]), (0, 0)), mode='constant')
        concate_image = np.concatenate([first_row, second_row, third_row_padded], axis=0)
        
    return concate_image
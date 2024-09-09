import numpy as np
import torch

@torch.no_grad()
def mask_onelabel_to_multilabel(masks: torch.Tensor) -> torch.Tensor:
    '''
    Convert a set of masks (integer of HxW) to a set of masks (bool of NxHxW)
    '''
    assert masks.ndim == 2, f"masks of shape {masks.shape} has a wrong number of dimensions"
    n_masks = masks.max().item() + 1 # -1 does not count
    mask_h, mask_w = masks.shape
    masks = masks.unsqueeze(0).expand(n_masks, mask_h, mask_w) # (N, H, W)
    masks_multilabel = masks == torch.arange(n_masks).unsqueeze(-1).unsqueeze(-1) # (N, H, W)

    return masks_multilabel # (N, H, W)

@torch.no_grad()
def xyxy_to_areas(xyxy: torch.Tensor) -> torch.Tensor:
    '''
    Convert a set of bounding boxes (in xyxy forms) to a set of areas

    Args:
        xyxy: tensor of bounding boxes (N, 4)

    Returns:
        areas: tensor of areas (N,)
    '''
    return (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

@torch.no_grad()
def expand_xyxy(xyxy: torch.Tensor, img_shape: tuple[int], scale: float = 1.1) -> torch.Tensor:
    '''
    Expand a set of bounding boxes (in xyxy forms) by a scale factor

    Args:
        xyxy: tensor of bounding boxes (N, 4)
        img_shape: tuple of image shape (H, W)
        scale: scale factor

    Returns:
        xyxy: tensor of bounding boxes (N, 4)
    '''
    if isinstance(xyxy, np.ndarray):
        xyxy = torch.from_numpy(xyxy)

    squeeze = False
    if xyxy.dim() == 1:
        squeeze = True
        xyxy = xyxy.unsqueeze(0)
    
    # Calculate the center of each box
    centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2

    # Calculate new half widths and heights
    hw = (xyxy[:, 2] - xyxy[:, 0]) * scale / 2
    hh = (xyxy[:, 3] - xyxy[:, 1]) * scale / 2

    # Create new boxes
    expanded_xyxy = torch.zeros_like(xyxy)
    expanded_xyxy[:, 0] = torch.clamp(centers[:, 0] - hw, min=0, max=img_shape[1])
    expanded_xyxy[:, 1] = torch.clamp(centers[:, 1] - hh, min=0, max=img_shape[0])
    expanded_xyxy[:, 2] = torch.clamp(centers[:, 0] + hw, min=0, max=img_shape[1])
    expanded_xyxy[:, 3] = torch.clamp(centers[:, 1] + hh, min=0, max=img_shape[0])

    if squeeze:
        expanded_xyxy = expanded_xyxy.squeeze(0)

    return expanded_xyxy


@torch.no_grad()
def masks_to_xyxy(masks: torch.Tensor, image_size: tuple = None) -> torch.Tensor:
    '''
    Convert set of masks to a set of bounding boxes (in xyxy forms)

    Args:
        masks: bool tensor of masks (N, H, W) or int tensor of masks (H, W)
            If bool, each mask is a binary mask of size (H, W)
            If int, the integer at each location is the label of the mask
        image_size: tuple of image size (H', W'). 
            If None, use the size of the masks. 
            Otherwise the returned bounding boxes will be scaled to image_size
    
    Returns:
        xyxy: tensor of bounding boxes (N, 4)
    '''
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
        
    if masks.dim() == 3:
        n_masks = masks.shape[0]
        mask_h, mask_w = masks.shape[1:]
    elif masks.dim() == 2:
        n_masks = masks.max().item() + 1
        mask_h, mask_w = masks.shape
    else:
        raise Exception(f"masks of shape {masks.shape} has a wrong number of dimensions")

    xyxy = torch.zeros((n_masks, 4), dtype=torch.float32)

    for mask_idx in range(n_masks):
        mask = masks[mask_idx] if masks.dim() == 3 else masks.eq(mask_idx)
        if mask.sum() == 0:
            continue
        
        # Find the bounding box coordinates
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]

        # If image_size is provided, scale the bounding box coordinates
        if image_size is not None:
            scale_y, scale_x = image_size[0] / float(mask_h), image_size[1] / float(mask_w)
            ymin, ymax = ymin * scale_y, ymax * scale_y
            xmin, xmax = xmin * scale_x, xmax * scale_x
            
        # Store the bounding box coordinates
        xyxy[mask_idx] = torch.tensor([xmin, ymin, xmax, ymax])
        
    return xyxy

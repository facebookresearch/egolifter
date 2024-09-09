import torch
from .gaussian_model import GaussianModel

class GsplatModel(GaussianModel):
    '''
    For compactibility with the gsplat implementation. 
    Reference: https://github.com/liruilong940607/gaussian-splatting/commit/258123ab8f8d52038da862c936bd413dc0b32e4d
    '''
    def add_densification_stats(self, viewspace_point_tensor, update_filter, width, height):
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]

        # Normalize the gradient to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5

        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
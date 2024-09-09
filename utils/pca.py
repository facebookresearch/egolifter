import copy
from pathlib import Path
import pickle

import numpy as np
from sklearn.decomposition import PCA
import torch
import tqdm

from scene import Scene
from model.vanilla import VanillaGaussian

class FeatPCA():
    def __init__(self, dim_pca:int) -> None:
        self.sampled_features = []
        self.dim_pca = dim_pca
        self.pca = None
        self.valid_indices_linear = None
        self.sampled_pca_max, self.sampled_pca_min = None, None
        self.n_sample_per_image = 1000

    def register_valid_mask(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.mask_linear = mask.reshape(-1) # (N,)
        self.valid_indices_linear = self.mask_linear.nonzero()[0]

    def add_sample(self, feat:np.ndarray) -> None:
        # Extract some features for dimension reduction
        assert feat.ndim == 2, "Input feature should be (N, D)"
        if self.valid_indices_linear is None:
            permute = np.random.permutation(feat.shape[0])
            indices = permute[:self.n_sample_per_image]
        else:
            permute = np.random.permutation(len(self.valid_indices_linear))
            indices = self.valid_indices_linear[permute[:self.n_sample_per_image]]
        self.sampled_features.append(feat[indices])

    def train(self) -> None:
        sampled_features = np.concatenate(self.sampled_features, axis=0)
        print(f"Features used for training: {sampled_features.shape}")
        self.pca = PCA(n_components = self.dim_pca)
        self.pca.fit(sampled_features)
        
        sampled_feaures_pca = self.pca.transform(sampled_features)
        self.sampled_pca_min = sampled_feaures_pca.min(axis=0) # (3,)
        self.sampled_pca_max = sampled_feaures_pca.max(axis=0) # (3,)

    def transform(self, feat:np.ndarray) -> None:
        assert self.pca is not None, "PCA model is not trained yet"
        assert feat.ndim == 2, "Input feature should be (N, D)"
        transformed = self.pca.transform(feat)
        transformed = (transformed - self.sampled_pca_min) / (self.sampled_pca_max - self.sampled_pca_min + 1e-6) # Normalize to [0, 1]
        transformed = np.clip(transformed, 0, 1)
        return transformed
    
    def save(self, path:Path) -> None:
        '''
        Save the underlying PCA object to a file
        '''
        with open(path, 'wb') as f:
            pickle.dump(self.pca, f)

def compute_feat_pca_from_renders(
        scene: Scene, 
        subset: str, 
        models: list[VanillaGaussian]
    ) -> list[FeatPCA]:
    loader = scene.get_data_loader(subset, shuffle=False, limit=100)
    pcas = [FeatPCA(dim_pca=3) for _ in models]
    
    for batch_idx, batch in tqdm.tqdm(enumerate(loader), total = len(loader)):
        subset = batch['subset'][0]
        viewpoint_cam = copy.deepcopy(scene.get_camera(batch['idx'].item(), subset=subset))
        viewpoint_cam.image_height = 512
        viewpoint_cam.image_width = 512

        with torch.no_grad():
            for i in range(len(models)):
                model = models[i]
                feat_img = model(viewpoint_cam, render_feature=True)['render_features'] # (D, H, W)
                feat_img = feat_img.permute(1, 2, 0).cpu().numpy().reshape(-1, feat_img.shape[0]) # (N, D)
                pcas[i].add_sample(feat_img)
        
    for i in range(len(models)):
        pcas[i].train()
    
    print("Computing the PCA model - Done")
    return pcas
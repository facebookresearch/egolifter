# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import Counter

import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from tqdm import tqdm

import hdbscan
import faiss

def bhattacharyya_distance_batch(p, q, r, s):
    '''
    Compute the Bhattacharyya distance between two paired set of Gaussian distributions
    in a vectorized manner, without explicit loops over the first dimension (N).

    Args:
        p: (N, 3) array of mean for the first distribution
        q: (N, 3) array of mean for the second distribution
        r: (N, 3, 3) array of covariance for the first distribution
        s: (N, 3, 3) array of covariance for the second distribution

    Returns:
        dist: (N, ) array of Bhattacharyya distance
    '''

    # Calculate the difference in means for each distribution
    mean_diff = p - q  # (N, 3)

    # Calculate the average covariance matrices
    cov_mean = (r + s) / 2  # (N, 3, 3)

    # Compute the inverse of the average covariance matrices
    inv_cov_mean = np.linalg.inv(cov_mean)  # (N, 3, 3)

    # Calculate the first term of the Bhattacharyya distance
    term1 = 1/8 * np.einsum('...i, ...ij, ...j', mean_diff, inv_cov_mean, mean_diff)  # (N, )

    # Calculate the log determinant of the average covariance matrix
    log_det_cov_mean = np.log(np.linalg.det(cov_mean))  # (N, )

    # Calculate the log determinant of the individual covariance matrices
    log_det_r = np.log(np.linalg.det(r))  # (N, )
    log_det_s = np.log(np.linalg.det(s))  # (N, )

    # Calculate the second term of the Bhattacharyya distance
    term2 = 0.5 * (log_det_cov_mean - 0.5 * (log_det_r + log_det_s))  # (N, )

    # Combine both terms to get the Bhattacharyya distance
    dist = term1 + term2  # (N, )

    return dist

def assign_noise_nearest_cluster(
    labeled_features: np.ndarray,
    labeled_labels: np.ndarray, 
    noise_features: np.ndarray, 
    labeled_cov: np.ndarray = None,
    noise_cov: np.ndarray = None,
    dist_thresh_l2: float = None,
    dist_thresh_bhatta: float = None,
):
    '''
    Assign each point in the noise set to the cluster label of its nearest point in the labeled set. 

    Args:
        labeled_features: (N, D) array of labeled features
        labeled_labels: (N, ) array of labeled labels
        noise_features: (M, D) array of noise features
        dist_thresh_l2: if the distance to the NN is larger than this, still consider it as noise. 

    Returns:
        noise_labels: (M, ) array of noise labels
    '''
    assert labeled_features.shape[1] == noise_features.shape[1]
    assert labeled_features.shape[0] == labeled_labels.shape[0]
    N, D = labeled_features.shape
    M = noise_features.shape[0]

    # build a faiss index
    index = faiss.IndexFlatL2(D)
    index.add(labeled_features.astype(np.float32))

    # search for the nearest neighbor
    dists, indices = index.search(noise_features.astype(np.float32), 1)
    noise_labels = labeled_labels[indices[:, 0]]

    if dist_thresh_l2 is not None:
        noise_labels[dists[:, 0] > dist_thresh_l2] = -1
        print(f"Assigning {np.sum(dists[:, 0] > dist_thresh_l2)} out of {len(noise_features)} noise points to noise after thresholding L2 distance of {dist_thresh_l2}. ")

    if dist_thresh_bhatta is not None:
        assert labeled_cov is not None
        assert noise_cov is not None
        assert labeled_cov.shape[0] == labeled_features.shape[0], f"{labeled_cov.shape} vs {labeled_features.shape}"
        assert noise_cov.shape[0] == noise_features.shape[0], f"{noise_cov.shape} vs {noise_features.shape}"

        dists_bhatta = bhattacharyya_distance_batch(
            noise_features, 
            labeled_features[indices[:, 0]], 
            noise_cov,
            labeled_cov[indices[:, 0]], 
        )

        noise_labels[dists_bhatta > dist_thresh_bhatta] = -1
        print(f"Assigning {np.sum(dists_bhatta > dist_thresh_bhatta)} out of {len(noise_features)} noise points to noise after thresholding Bhatta distance of {dist_thresh_bhatta}. ")

    return noise_labels


def gaussian_cluster_instance(
    args: argparse.Namespace,
    features: np.ndarray, 
    xyz: np.ndarray,
    covariance: np.ndarray,
    opacity: np.ndarray,
    point_labels_cluster_init: np.ndarray = None,
):
    if args.seman_dim < args.feat_dim:
        print("Using contrastive features for clustering...")
        features_inst = features[:, args.seman_dim:]
        if args.spatial_weight > 0:
            print(f"Also using spatial positions with weight {args.spatial_weight} for clustering...")
            features_inst = np.concatenate([features_inst, xyz * args.spatial_weight], axis=1)
    else:
        features_inst = xyz
        print("Using only spatial positions for clustering...")

    # Perform clustering on opaque points
    # Then assign transparent points to the nearest cluster
    if args.opacity_thresh > 0:
        opaque_mask = opacity > args.opacity_thresh
        transparent_mask = opacity <= args.opacity_thresh
        features_inst_opaque = features_inst[opaque_mask]
        features_inst_transparent = features_inst[transparent_mask]
        print(f"number of opaque points: {len(features_inst_opaque)}, transparent points: {len(features_inst_transparent)}")
    else:
        opaque_mask = np.ones(len(features_inst), dtype=bool)
        transparent_mask = np.zeros(len(features_inst), dtype=bool)
        features_inst_opaque = features_inst
    
    # Perform the actual clustering
    if args.clusterer == "kmeans":
        print("Running K-Means clustering")
        clusterer = KMeans(
            n_clusters=args.num_clusters,
            random_state=0,
            n_init='auto',
        ).fit(features_inst_opaque)
        # point_labels_cluster = clusterer.labels_
        print("KMeans clustering done")
    elif args.clusterer == "hdbscan":
        print("Running HDBSCAN clustering")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size = args.hdbscan_min_cluster_size,
            min_samples = args.hdbscan_min_samples,
            cluster_selection_epsilon = args.hdbcsan_epsilon,
        ).fit(features_inst_opaque)
    else:
        raise ValueError(f"Unknown clusterer {args.clusterer}")

    # Assign the clusters back to the opacque points
    # Now -1 means either the point is transparent or it is considered as noise by the clustering algorithm
    point_labels_cluster = np.ones(len(features_inst), dtype=int) * -1
    point_labels_cluster[opaque_mask] = clusterer.labels_
    point_labels_cluster_init = point_labels_cluster.copy()

    cluster_noise_mask = np.zeros(len(features_inst), dtype=bool)
    cluster_noise_mask[opaque_mask] = clusterer.labels_ == -1

    # Reassign clustered noise to nearest cluster
    labeled_features = xyz[point_labels_cluster != -1]
    labeled_labels = point_labels_cluster[point_labels_cluster != -1]
    labeled_cov = covariance[point_labels_cluster != -1]
    noise_features = xyz[point_labels_cluster == -1]
    noise_cov = covariance[point_labels_cluster == -1]

    noise_labels = assign_noise_nearest_cluster(
        labeled_features, 
        labeled_labels, 
        noise_features, 
        labeled_cov=labeled_cov,
        noise_cov=noise_cov,
        dist_thresh_l2=args.reassign_dist_thresh,
        dist_thresh_bhatta=args.reassign_bhatta_dist_thresh,
    )
    point_labels_cluster[point_labels_cluster == -1] = noise_labels

    args.num_clusters = len(np.unique(point_labels_cluster))
    print(f"The clustering algorithm found {args.num_clusters} clusters. Overriding...")
    
    # Denoise the clusters
    if args.denoise_dbscan_eps > 0:
        print("Denoising each cluster with DBSCAN")
        point_labels_cluster_denoised = point_labels_cluster.copy()
        for l in tqdm(np.unique(point_labels_cluster)):
            mask = (point_labels_cluster == l).nonzero()[0]
            pcd_cluster = o3d.geometry.PointCloud()
            pcd_cluster.points = o3d.utility.Vector3dVector(xyz[mask])
            labels = pcd_cluster.cluster_dbscan(eps=args.denoise_dbscan_eps, min_points=args.denoise_dbscan_min_points)
            labels = np.array(labels)
            counter = Counter(labels)
            point_labels_cluster_denoised[mask[labels == -1]] =-1
            for c in counter.keys():
                if c == -1:
                    continue
                point_labels_cluster_denoised[mask[labels == c]] = point_labels_cluster_denoised.max() + 1
        args.num_clusters = point_labels_cluster_denoised.max() # includes the noise points
        print(f"DBSCAN denoising done. Found {args.num_clusters} clusters.")
    else:
        point_labels_cluster_denoised = point_labels_cluster


    return point_labels_cluster_denoised, point_labels_cluster_init, args.num_clusters, transparent_mask, cluster_noise_mask
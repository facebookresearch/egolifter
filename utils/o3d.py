# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import warnings
from collections import Counter

import imageio
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import torch

def pcd_mask_color(pcd_original, mask, color):
    if isinstance(color, list):
        color = np.asarray(color)
    if color.ndim == 1:
        color = color[None, :]
    pcd = copy.deepcopy(pcd_original)
    colors = np.asarray(pcd.colors)
    colors[mask] = color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def better_camera_frustum(camera_pose, img_h, img_w, scale=3.0, color=[0, 0, 1]):
    # Convert camera pose tensor to numpy array
    if isinstance(camera_pose, torch.Tensor):
        camera_pose = camera_pose.numpy()
    
    # Define near and far distance (adjust these as needed)
    near = scale * 0.1
    far = scale * 1.0
    
    # Define frustum dimensions at the near plane (replace with appropriate values)
    frustum_h = near
    frustum_w = frustum_h * img_w / img_h  # Set frustum width based on its height and the image aspect ratio
    
    # Compute the 8 points that define the frustum
    points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                u = x * (frustum_w // 2 if z == -1 else frustum_w * far / near)
                v = y * (frustum_h // 2 if z == -1 else frustum_h * far / near)
                d = near if z == -1 else far # Negate depth here
                # d = -near if z == -1 else -far # Negate depth here
                point = np.array([u, v, d, 1]).reshape(-1, 1)
                transformed_point = (camera_pose @ point).ravel()[:3]
                # transformed_point[0] *= -1  # Flip X-coordinate
                points.append(transformed_point) # Using camera pose directly
                # points.append((camera_pose_np @ point).ravel()[:3]) # Using camera pose directly
    
    # Create lines that connect the 8 points
    lines = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [3, 7], [2, 6]]
    
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return frustum

def visualize_pcd_with_cameras(
    geometries, camera_poses,
    window_width:int = 2560,
    window_height:int = 1440,
    fov: int = 60,
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height)
    vis_ctrl = vis.get_view_control()
    vis_ctrl.change_field_of_view(step=fov - vis_ctrl.get_field_of_view())
    for g in geometries:
        vis.add_geometry(g)
    cmap = plt.get_cmap('jet')
    for i, pose in enumerate(camera_poses):
        color = cmap(float(i) / len(camera_poses))
        frustum = better_camera_frustum(pose, 300, 300, scale=0.15, color=color[:3])
        vis.add_geometry(frustum)
    # Draw a coordinate system at the origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    vis.run()
    vis.destroy_window()

def o3d_visualize_capture_trajectory(
    geometries: list[o3d.geometry.Geometry],
    camera_extrinsics: np.ndarray = None,
    window_width:int = 1408,
    window_height:int = 1408,
    field_of_view: int = 60,
    save_folder: str = None,
    line_width: float = None,
) -> list[np.ndarray]:
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name = "Open3D Animation",
        width = window_width,
        height = window_height,
    )

    for g in geometries:
        vis.add_geometry(g)

    view_ctrl = vis.get_view_control()
    view_ctrl.change_field_of_view(field_of_view - view_ctrl.get_field_of_view()) # By default, the field of view is 60

    if line_width is not None:
        print("Setting line width probably does not work for open3d.")
        vis.get_render_option().line_width = line_width

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
    
    if camera_extrinsics is None:
        # Simply run the visualization. 
        vis.run()
    else:
        # Capture geometries following the camera poses
        captured_frames = []
        for idx_pose, pose in enumerate(camera_extrinsics):
            # Move to the next pose
            camera_param = view_ctrl.convert_to_pinhole_camera_parameters()
            camera_param.extrinsic = pose
            view_ctrl.convert_from_pinhole_camera_parameters(camera_param, allow_arbitrary=True)
            vis.poll_events()
            vis.update_renderer()

            # Capture the image
            render_rgb = vis.capture_screen_float_buffer(False)
            render_rgb = np.asarray(render_rgb)
            render_rgb = (render_rgb * 255).astype(np.uint8)

            if save_folder is not None:
                # To avoid OOM, only save, not accumulate
                save_path = os.path.join(save_folder, f"{idx_pose:06d}.jpg")
                imageio.imwrite(save_path, render_rgb)
            else:
                captured_frames.append(render_rgb)

    vis.destroy_window()

    return captured_frames

def visualize_get_extrinsic(
    geometries: list[o3d.geometry.Geometry],
    window_width:int = 2560, 
    window_height:int = 1440,
    vis_bbox: bool = False,
    init_extrinsic: np.ndarray = None,
    fov: int = 90,
) -> np.ndarray:
    """
    Visualize the given geometries and get the camera pose
        
    Args:
    - geometries (list of o3d.geometry.Geometry): The list of geometries to visualize
    - window_width (int): The width of the visualization window
    - window_height (int): The height of the visualization window
    - vis_bbox (bool): Whether to visualize the bounding box of the point cloud
    - init_extrinsic (np.ndarray): The initial camera pose
            
    Returns:
    - np.ndarray: The camera pose
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=window_width, 
        height=window_height,
    )
    for g in geometries:
        vis.add_geometry(g)

        if isinstance(g, o3d.geometry.PointCloud) and vis_bbox:
            bbox = g.get_axis_aligned_bounding_box()
            # bbox = g.get_oriented_bounding_box(robust=True)
            bbox.color = (1, 0, 0)
            vis.add_geometry(bbox)
            
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=fov - ctr.get_field_of_view())
    if init_extrinsic is not None:
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = init_extrinsic
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

    vis.run()

    # Get the current camera pose
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    
    vis.destroy_window()

    return param.extrinsic

def visualize_get_extrinsic_click(
    pcd: o3d.geometry.PointCloud,
    window_width:int = 2560, 
    window_height:int = 1440,
    init_extrinsic: np.ndarray = None,
    fov: int = 60,
):
    vis = o3d.visualization.VisualizerWithEditing()
    # vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=window_width, 
        height=window_height,
    )
    vis.add_geometry(pcd)
     
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=fov - ctr.get_field_of_view())
    if init_extrinsic is not None:
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = init_extrinsic
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

    print("")
    print("==> Please pick points using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("   Only the last point picking will be considered")
    print("==> After picking points, press q to close the window")
    vis.run()

    picked_indices = np.array(vis.get_picked_points())

    # Get the current camera pose
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    
    vis.destroy_window()
    return param.extrinsic, picked_indices

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10, largest_only=True) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        if largest_only:
            # Find the label of the largest cluster
            most_common_label, _ = counter.most_common(1)[0]
            # Create mask for points in the largest cluster
            mask = pcd_clusters == most_common_label
        else:
            mask = pcd_clusters != -1

        # Apply mask
        largest_cluster_points = obj_points[mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            warnings.warn("Selected cluster too small. Returning the original point cloud. ")
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        if len(obj_colors) > 0:
            largest_cluster_colors = obj_colors[mask]
            largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
    else:
        warnings.warn("All points are treated as noise. Returning the original point cloud.")
        
    return pcd

def get_bbox_geometries_by_label(xyz, label, picked_cluster, palette, denoise_eps=0.05):
    # Also create bounding boxes for selected clusters
    bbox_meshes = []
    
    # for cluster_id in range(label.max()):
    for cluster_id in picked_cluster:
        mask_cluster = label == cluster_id
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(xyz[mask_cluster])
        pcd_cluster = pcd_denoise_dbscan(pcd_cluster, eps=denoise_eps, min_points=15, largest_only=False)
        bbox_points = compute_bounding_box_gravity(np.asarray(pcd_cluster.points), np.asarray([0.0, 0.0, 1.0]))
        bbox_lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]

        bbox_mesh = LineMesh(bbox_points, bbox_lines, colors=palette[cluster_id], radius=0.006)
        bbox_meshes.append(bbox_mesh)
        
    bbox_geometries = []
    for bbox_mesh in bbox_meshes:
        bbox_geometries.extend(bbox_mesh.cylinder_segments)
    
    return bbox_geometries

def interactive_merge_cluster(xyz:np.ndarray, labels:np.ndarray, palette:np.ndarray, init_extrinsic:np.ndarray=None):
    # Interactive merge clusters using open3d.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(palette[labels])
    labels_original = labels.copy()
    
    camera_extrinsic = init_extrinsic
    
    while True:
        print("Please pick the clusters you want to merge. ")
        camera_extrinsic, picked_indices = visualize_get_extrinsic_click(
            pcd,
            window_width=1920, 
            window_height=1920,
            init_extrinsic=camera_extrinsic,
        )

        if len(picked_indices) == 0:
            print("No point is picked. Finish merging...")
            break
        
        if len(picked_indices) == 1:
            print("Only one point is picked. Please pick more points.")
            continue
        
        # Merge the picked clusters
        picked_labels = labels[picked_indices]
        picked_labels = np.unique(picked_labels)
        print(f"Merging clusters {picked_labels}")
        labels[np.isin(labels, picked_labels)] = picked_labels.max()
        
        pcd.colors = o3d.utility.Vector3dVector(palette[labels])
    
    satisfied = input("Are you satisfied with the merging? [y/n] ")
    if satisfied.lower().strip() == "y":
        return labels
    elif satisfied.lower().strip() == "n":
        print("Let's start over again.")
        return interactive_merge_cluster(xyz, labels_original, palette, init_extrinsic)
    else: 
        raise ValueError("Please input y or n.")


# Copied from https://github.com/isl-org/Open3D/pull/738
def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                # cylinder_segment = cylinder_segment.rotate(
                #     R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder, reset_bounding_box=False)

    def remove_line(self, vis):
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder, reset_bounding_box=False)

def compute_bounding_box_gravity(points, gravity_direction):
    # Normalize the gravity_direction
    gravity_direction = gravity_direction / np.linalg.norm(gravity_direction)
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    # 1. Projecting centered_points onto the gravity direction
    projections_gravity = np.dot(centered_points, gravity_direction)
    min_along_gravity = projections_gravity.min()
    max_along_gravity = projections_gravity.max()
    
    # 2. Projecting centered_points to the plane orthogonal to gravity
    points_orthogonal = centered_points - np.outer(projections_gravity, gravity_direction)
    
    # 3. PCA for 2D Bounding Box
    pca = PCA(n_components=2)
    pca.fit(points_orthogonal)
    
    # Corners of the 2D bounding box
    xyxy = np.array([
        points_orthogonal.dot(pca.components_[0]).min() * pca.components_[0],
        points_orthogonal.dot(pca.components_[1]).min() * pca.components_[1],
        points_orthogonal.dot(pca.components_[0]).max() * pca.components_[0],
        points_orthogonal.dot(pca.components_[1]).max() * pca.components_[1],
    ])
    corners_2D = np.array([
        xyxy[0] + xyxy[1],
        xyxy[0] + xyxy[3],
        xyxy[2] + xyxy[3],
        xyxy[2] + xyxy[1],
    ])
    
    # 4. Combining the 2D Bounding Box with the Range along Gravity
    corners_3D = []
    for d in [min_along_gravity, max_along_gravity]:
        for corner in corners_2D:
            point = corner + d * gravity_direction
            corners_3D.append(point)

    corners_3D = np.array(corners_3D)
    corners_3D += center[None, :]
    
    return corners_3D

if __name__ == "__main__":
    def generate_ellipsoid(a, b, c, num_points=10000):
        phi = np.random.uniform(0, np.pi, num_points)
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        points = np.zeros((num_points, 3))
        points[:, 0] = a * np.sin(phi) * np.cos(theta)
        points[:, 1] = b * np.sin(phi) * np.sin(theta)
        points[:, 2] = c * np.cos(phi)
        return points

    def visualize_point_cloud_with_bounding_box(points, corners):
        # Convert points to open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Create bounding box lines using the corners
        lines = [
            # [0, 1], [0, 2], [0, 3], 
            # [4, 5], [4, 6], [4, 7], 
            # [3, 5], [3, 6], [1, 6], [1, 7], [2, 5], [2, 7]
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Draw coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)

        # Visualize
        o3d.visualization.draw_geometries([pcd, line_set, coordinate_frame])

    # Sample usage:
    ellipsoid_points = generate_ellipsoid(5, 3, 10)
    ellipsoid_points = ellipsoid_points[::30]
    # Rotate the ellipsoid by a random rotation
    # rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(0, 2 * np.pi, 3))
    rotation = o3d.geometry.get_rotation_matrix_from_xyz([1.0, 1.0, 0.8])
    ellipsoid_points = np.dot(ellipsoid_points, rotation)

    direction = np.array([0.0, 0.0, 1.0])
    corners = compute_bounding_box_gravity(ellipsoid_points, direction)
    visualize_point_cloud_with_bounding_box(ellipsoid_points, corners)
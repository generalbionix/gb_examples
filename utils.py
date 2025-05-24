import numpy as np
from typing import Tuple
import open3d as o3d

def compute_mask_center_of_mass(mask: np.ndarray) -> Tuple[int, int]:
    """
    Compute the center of mass of a binary segmentation mask.

    Args:
        mask (np.ndarray): A binary mask of shape (H, W) where pixels belonging to 
                          the object are 1 (or True) and background pixels are 0 (or False).

    Returns:
        tuple: (x, y) coordinates of the center of mass in pixel coordinates.
    """
    # Ensure the mask is a binary mask
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    # Get the indices of the pixels that are part of the mask
    y_indices, x_indices = np.nonzero(mask)  # y_indices and x_indices are the coordinates of the true pixels

    # Calculate the total number of pixels in the mask
    total_pixels = len(x_indices)

    # If there are no pixels in the mask, return None or appropriate value
    if total_pixels == 0:
        return None

    # Compute the center of mass
    center_x = np.sum(x_indices) / total_pixels
    center_y = np.sum(y_indices) / total_pixels

    return int(center_x), int(center_y)



def downsample_pcd(pcd: o3d.geometry.PointCloud, down_sample: int) -> o3d.geometry.PointCloud:
    """"
    Uniformly downsample the point cloud for network transfer.
    Args:
        pcd: o3d.geometry.PointCloud: Orignal point cloud
        down_sample: int
    Returns:
        o3d.geometry.PointCloud: Downsampled point cloud.
    """
    pcd_ds = pcd.uniform_down_sample(down_sample) # Downsample for faster processing
    return pcd_ds


def upsample_pcd(pcd_ds: o3d.geometry.PointCloud, pcd_full: o3d.geometry.PointCloud, up_sample: int) -> o3d.geometry.PointCloud:
    """
    Upsample the downsampled point cloud based on the original pointcloud using nearest neighbor search.
    Args:
        pcd_ds: o3d.geometry.PointCloud: Downsampled point cloud.
        pcd_full: o3d.geometry.PointCloud: Orignal point cloud.
        up_sample: int
    Returns:
        pcd_us: o3d.geometry.PointCloud: Upsampled point cloud.
    """
    k = up_sample 
    voxel = pcd_full.get_max_bound() - pcd_full.get_min_bound()
    avg_spacing = (np.linalg.norm(voxel) / np.cbrt(len(pcd_full.points)))  # ~mean NN spacing
    radius = k * 0.05 * avg_spacing       # anything inside this radius was probably dropped
    kdtree_full = o3d.geometry.KDTreeFlann(pcd_full)
    hits = set()
    for q in pcd_ds.points:                 # each “query” point
        _, idxs, _ = kdtree_full.search_radius_vector_3d(q, radius)
        hits.update(idxs)                       # collect everything nearby
    pcd_us = pcd_full.select_by_index(list(hits))
    return pcd_us
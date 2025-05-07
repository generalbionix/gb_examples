import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import open3d as o3d

import cameras

ROTATION = cameras.RealSenseD435.CONFIG[0]["rotation"]  # quaternian
TRANSLATION = cameras.RealSenseD435.CONFIG[0]["position"]


def make_transform_mat() -> np.array:
    """
    Make a transformation matrix from the camera pose.

    Returns:
        np.array: Transformation matrix
    """
    rotation_mat = R.from_quat(ROTATION).as_matrix()
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = rotation_mat
    transform_mat[:3, 3] = TRANSLATION
    return transform_mat


def transform_cam_to_rob(rotation_matrix : np.array, translation : np.array) -> Tuple[np.array, np.array]:
    """
    Transform gripper pose using ground truth camera pose.

    Args:
        gg: GraspGroup instance with a single grasp

    Returns:
        Any: Transformed GraspGroup instance
    """
    gripper_pose = np.eye(4)
    gripper_pose[:3, :3] = rotation_matrix
    gripper_pose[:3, 3] = translation
    transform_mat = make_transform_mat()
    x180 = R.from_euler("xyz", [-180, 0, 0], degrees=True).as_matrix()
    transform_180 = np.eye(4)
    transform_180[:3, :3] = x180
    res1 = transform_180 @ gripper_pose
    res = transform_mat @ res1
    transformed_rotation = res[:3, :3]
    transformed_translation = res[:3, 3]
    return transformed_rotation.reshape((3, 3)), transformed_translation.reshape((3,))


def transform_pcd_cam_to_rob(pcd : o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Transform a point cloud from camera frame to robot frame.
    """
    transform_mat = make_transform_mat()
    pcd.transform(transform_mat)
    return pcd
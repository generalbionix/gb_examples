import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from typing import List
from client import Grasp
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
    Args:
        pcd: o3d.geometry.PointCloud: Point cloud in camera frame.
    Returns:
        o3d.geometry.PointCloud: Transformed point cloud in robot frame.
    """
    transform_mat = make_transform_mat()
    pcd.transform(transform_mat)
    return pcd


def transform_grasps_inv(grasps: List[Grasp]) -> List[Grasp]:
    """
    For visualization the grasp definitions are slightly different than what PyBullet uses,
    so this function is to compensate for that.
    We rotation the grasp by -90deg in the gripper frame and move it by 15cm along the gripper Z axis.
    
    Args:
        grasps: List[Grasp]: List of grasps in robot frame.
    Returns:
        List[Grasp]: List of grasps in camera frame.
    """
    for i in range(len(grasps)):
        rz = -np.pi / 2
        rot = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )
        # Apply the rotation to every grasp (broadcasted matmul)
        grasps[i].rotation = np.matmul(np.array(grasps[i].rotation), rot).tolist()

        approach_distance = 0.15  # [m] total travel along +Z gripper axis
        # Extract rotation matrices (N, 3, 3) and +Z axes expressed in the robot frame (N, 3)
        rot_mats = np.array(grasps[i].rotation)
        z_axes = rot_mats[
            :, 2
        ]  # third column is the gripper +Z-axis in robot frame
        z_axes /= np.linalg.norm(z_axes)  # normalise
        # Translate grasp origins along the +Z axis
        grasps[i].translation = (np.array(grasps[i].translation) - approach_distance * z_axes).tolist()
    return grasps
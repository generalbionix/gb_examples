import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from typing import List
from client import Grasp
import cameras
from dataclasses import dataclass

from compute_transform_mat import REAL_ROBOT_TRANSFORM_PATH, REAL_ROBOT_SCALING_FACTOR_PATH

ROTATION = cameras.RealSenseD435.CONFIG[0]["rotation"]  # quaternian
TRANSLATION = cameras.RealSenseD435.CONFIG[0]["position"]

SO100_X_GRASP_SHIFT = 0
SO100_Y_GRASP_SHIFT = 0.025
SO100_Z_GRASP_SHIFT = -0.1


@dataclass
class GBRobotConfig:
    x_grasp_shift: float = 0
    y_grasp_shift: float = 0
    z_grasp_shift: float = 0
    robot_type: str = None


def make_robot_config(robot_type : str) -> GBRobotConfig:
    """
    Make a robot config object describing the grasp offsets for a given robot.

    Args:
        robot_type (str): The type of robot. Options: "so100"
    Returns:
        GBRobotConfig: The robot config object.
    """
    if robot_type == "so100":
        return GBRobotConfig(x_grasp_shift=SO100_X_GRASP_SHIFT, y_grasp_shift=SO100_Y_GRASP_SHIFT, z_grasp_shift=SO100_Z_GRASP_SHIFT, robot_type="so100")
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")


def make_transform_mat(real_robot : bool) -> np.array:
    """
    Make a transformation matrix from the camera pose.
    
    Args:
        real_robot (bool): Whether to use transform matrix from real robot or ground truth from pybullet sim
    Returns:
        np.array: Transformation matrix
    """
    if real_robot:
        transform_mat = np.load(
            REAL_ROBOT_TRANSFORM_PATH,
            allow_pickle=True,
        )
        transform_mat[:3, 3] /= np.load(
            REAL_ROBOT_SCALING_FACTOR_PATH,
            allow_pickle=True,
        )
        return transform_mat
    else:
        rotation_mat = R.from_quat(ROTATION).as_matrix()
        transform_mat = np.eye(4)
        transform_mat[:3, :3] = rotation_mat
        transform_mat[:3, 3] = TRANSLATION
        return transform_mat


def transform_cam_to_rob(rotation_matrix : np.array, translation : np.array, real_robot : bool) -> Tuple[np.array, np.array]:
    """
    Transform gripper pose using ground truth camera pose.

    Args:
        rotation_matrix: np.array: Rotation matrix of the gripper in camera frame
        translation: np.array: Translation vector of the gripper in camera frame
        real_robot: bool: Whether to use transform matrix from real robot or ground truth from pybullet sim

    Returns:
        Tuple[np.array, np.array]: Transformed rotation matrix and translation vector in robot frame
    """
    gripper_pose = np.eye(4)
    gripper_pose[:3, :3] = rotation_matrix
    gripper_pose[:3, 3] = translation
    transform_mat = make_transform_mat(real_robot)
    transform_mat = make_transform_mat(real_robot)
    x180 = R.from_euler("xyz", [-180, 0, 0], degrees=True).as_matrix()
    transform_180 = np.eye(4)
    transform_180[:3, :3] = x180
    res1 = transform_180 @ gripper_pose
    res = transform_mat @ res1
    transformed_rotation = res[:3, :3]
    transformed_translation = res[:3, 3]
    return transformed_rotation.reshape((3, 3)), transformed_translation.reshape((3,))


def transform_pcd_cam_to_rob(pcd : o3d.geometry.PointCloud, real_robot : bool) -> o3d.geometry.PointCloud:
    """
    Transform a point cloud from camera frame to robot frame.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud in camera frame.
        real_robot (bool): Whether to use transform matrix from real robot or ground truth from pybullet sim
    Returns:
        o3d.geometry.PointCloud: Transformed point cloud in robot frame.
    """
    transform_mat = make_transform_mat(real_robot)
    pcd.transform(transform_mat)
    return pcd


def transform_grasps_inv(grasps: List[Grasp]) -> List[Grasp]:
    """
    For visualization the grasp definitions are slightly different than what PyBullet uses,
    so this function is to compensate for that.
    We rotation the grasp by -90deg in the gripper frame and move it by 15cm along the gripper Z axis.
    
    Args:
        grasps (List[Grasp]): List of grasps in robot frame.
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


def change_to_robot_coord_system(rot_mat: np.ndarray, robot_config : GBRobotConfig) -> np.ndarray:
    """
    The SO100 arm URDF has a different coordinate system than the grasp prediction service:
    - There is a constant (-90, 0, 90) rotation
    - The x and y axes are swapped
    - The resulting y axis is inverted.
    Placeholder for other robots with different coordinate systems.
    
    Args:
        rot_mat (np.ndarray): The rotation matrix of the grasp.
    Returns:
        np.ndarray: The rotation matrix of the grasp in the SO-100 coordinate system.
    """
    # TODO: fix the URDF so this is not needed
    if robot_config.robot_type == "so100":
        xyz = R.from_matrix(rot_mat).as_euler("xyz", degrees=True).tolist()
        robot_rot = R.from_euler(
            "xyz", [-90 + xyz[1], 0 - xyz[0], 90 + xyz[2]], degrees=True
        ).as_matrix()
        return robot_rot
    else:
        return rot_mat


def _offset_grasp(robot_config : GBRobotConfig, grasp : Grasp, robot_rot : np.ndarray) -> Grasp:
    """
    Offset the grasp by the amount specified in the robot config.

    Args:
        robot_config (GBRobotConfig): The robot config object.
        grasp (Grasp): The grasp to offset.
        robot_rot (np.ndarray): The rotation matrix of the grasp in the robot frame.
    Returns:
        Grasp: The offset grasp.
    """
    rot_mat = np.array(grasp.rotation)
    x_axis = rot_mat[
        :, 0
    ]  # first column is the gripper +X-axis in robot frame
    x_axis /= np.linalg.norm(x_axis)  # normalise
    y_axis = rot_mat[
        :, 1
    ]  # second column is the gripper +Y-axis in robot frame
    y_axis /= np.linalg.norm(y_axis)  # normalise
    z_axes = rot_mat[
        :, 2
    ]  # third column is the gripper +Z-axis in robot frame
    z_axes /= np.linalg.norm(z_axes)  # normalise
    # Translate grasp origins along the +Z axis
    translation_moved = (np.array(grasp.translation) + robot_config.z_grasp_shift * z_axes + robot_config.y_grasp_shift * y_axis + robot_config.x_grasp_shift * x_axis).tolist()
    new_grasp = Grasp(translation=translation_moved, rotation=robot_rot.tolist())
    return new_grasp


def grasp_service_to_robot_format(grasps: List[Grasp], robot_config : GBRobotConfig) -> List[Grasp]:
    """Convert grasps from grasp service to robot grasp format.

    To make the grasp prediction service work for the robot, we must 
    make a couple of adjustments to the grasp predictions:
    - Potential coordinate system change
    - Translate the grasp point

    Args:
        grasps (List[Grasp]): List of grasps in PIPER format.

    Returns:
        List[Grasp]: List of grasps in SO-100 format.
    """
    res = []
    for grasp in grasps:
        try:
            robot_rot = change_to_robot_coord_system(np.array(grasp.rotation), robot_config)
        except ValueError:
            print("ValueError: Non-positive determinant (left-handed or null coordinate frame) in rotation matrix")
            continue
        offset_grasp = _offset_grasp(robot_config, grasp, robot_rot=robot_rot)
        res.append(offset_grasp)
    return res
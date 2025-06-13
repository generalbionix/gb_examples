"""
This module computes transformation matrices between camera space and robot space.
Uses SVD to compute the optimal rigid transformation (rotation + translation) given 
a set of corresponding points in camera and robot space.
Saves the resulting transform matrix and scaling factor to config/transform_mat.npy and config/scaling_factor.npy.
"""

from typing import Any
import numpy as np

REAL_ROBOT_TRANSFORM_PATH = "config/transform_mat.npy"
REAL_ROBOT_SCALING_FACTOR_PATH = "config/scaling_factor.npy"


def compute_transform_matrix(
    robot_points: np.ndarray, camera_points: np.ndarray
) -> np.ndarray:
    """
    Compute the transformation matrix that maps points from camera space to robot space.
    Uses SVD to find the optimal rigid transformation (rotation + translation).

    Args:
        robot_points: Nx3 array of points in robot space (mm)
        camera_points: Nx3 array of points in camera space (m)

    Returns:
        4x4 transformation matrix that converts from camera space (m) to robot space (mm)
    """
    # Ensure we're working with numpy arrays
    robot_points = np.array(robot_points)
    camera_points = np.array(camera_points)

    # Center the point clouds
    robot_centroid = np.mean(robot_points, axis=0)
    camera_centroid = np.mean(camera_points, axis=0)

    robot_centered = robot_points - robot_centroid
    camera_centered = camera_points - camera_centroid

    # Compute the covariance matrix
    H = camera_centered.T @ robot_centered

    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Handle special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = robot_centroid - R @ camera_centroid

    # Create 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t

    return transform_matrix


def get_scaling_factor(camera_poses: np.ndarray, robot_poses: np.ndarray) -> Any:
    """
    Compute the scaling factor between camera and robot poses.

    Args:
        camera_poses: Nx3 array of camera poses (in meters)
        robot_poses: Nx3 array of robot poses (in millimeters)
    Returns:
        float: scaling factor
    """
    dist_ratios = []

    for i in range(len(camera_poses)):
        for j in range(len(camera_poses)):
            if i != j:
                dist_cam = np.linalg.norm(camera_poses[i] - camera_poses[j])
                dist_robot = np.linalg.norm(robot_poses[i][:3] - robot_poses[j][:3])
                dist_ratios.append(dist_robot / dist_cam)

    return np.mean(dist_ratios)


def scale_camera_poses(camera_poses: np.ndarray, scaling_factor: float) -> np.ndarray:
    """
    Scale the camera poses by the scaling factor.

    Args:
        camera_poses: Nx3 array of camera poses (in meters)
        scaling_factor: float
    """
    return camera_poses * scaling_factor


def calibrate(camera_poses: np.ndarray, robot_poses: np.ndarray):
    """
    Wrapper around compute_transform_matrix() that saves the results and computes and prints the calibration error.

    Args:
        camera_poses (np.ndarray): Nx3 array of camera poses (in meters)
        robot_poses (np.ndarray): Nx3 array of robot poses (in meters)
    """

    scaling_factor = get_scaling_factor(camera_poses, robot_poses)
    np.save(REAL_ROBOT_SCALING_FACTOR_PATH, scaling_factor)

    print(f"Scaling factor: {scaling_factor}")
    camera_poses = scale_camera_poses(camera_poses, scaling_factor)

    # Compute the transformation matrix
    transform_matrix = compute_transform_matrix(robot_poses, camera_poses)

    # Save the transformation matrix
    np.save(REAL_ROBOT_TRANSFORM_PATH, transform_matrix)
    print(f"Transformation matrix saved to {REAL_ROBOT_TRANSFORM_PATH}")

    # Print the transformation matrix for reference
    print("\nTransformation Matrix:")
    print(transform_matrix)

    # Test the transformation on the input points
    print("\nValidation - Transforming camera points to robot space:")
    for i in range(len(camera_poses)):
        print(f"Point {i + 1}:")
        print(f"  Camera (m): {camera_poses[i]}")

        # Make homogeneous point (in meters as original)
        camera_point_homogeneous = np.append(camera_poses[i], 1)

        # Apply transformation (inherently handles m to mm conversion)
        transformed_point = transform_matrix @ camera_point_homogeneous

        # Original robot point (just xyz)
        orig_robot_point = robot_poses[i]

        # Calculate error
        error = np.linalg.norm(transformed_point[:3] - orig_robot_point)
        error_percent = (
            (error / np.linalg.norm(orig_robot_point)) * 100
            if np.linalg.norm(orig_robot_point) > 0
            else 0
        )

        print(f"  Robot (original, m): {orig_robot_point}")
        print(f"  Robot (transformed, m): {transformed_point[:3]}")
        print(f"  Error: {error:.2f} m ")


"""Camera configs."""

import numpy as np
import pybullet as p


class RealSenseD435:
    """Default configuration with 3 RealSense RGB-D cameras.
    https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html
    camera_image_size = (480, 640)
    camera_fov_w = 69.4  # horizontal field of view, width of image
    camera_focal_length = (float(camera_image_size[1]) / 2) / np.tan((np.pi * camera_fov_w / 180) / 2)
    camera_focal_length = 462.14
    """

    # Mimic RealSense D435 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = np.array([[462.14, 0, 320], [0, 462.14, 240], [0, 0, 1]])

    # Set default camera poses.
    # relative to the manipulator
    front_position = (0.3, 0, 0.5)
    front_rotation = (0, np.pi, np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)

    # Default camera configs.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": left_position,
            "rotation": left_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": right_position,
            "rotation": right_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
    ]
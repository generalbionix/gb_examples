from typing import Any, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pybullet as pb
import pybullet_data
import open3d as o3d
import time

import cameras
from transform import transform_cam_to_rob
from client import Grasp


EEF_IDX = 9  

# Object properties constants
DUCK_ORIENTATION = [np.pi / 2, 0, 0]
DUCK_SCALING = 0.7
TRAY_ORIENTATION = [0, 0, 0]
TRAY_SCALING = 0.3
TRAY_POS = [0.3, -0.3, 0.0]
CUBE_ORIENTATION = [0, 0, 0]
CUBE_SCALING = 0.7
SPHERE_ORIENTATION = [0, 0, 0]
SPHERE_SCALING = 0.05
SPHERE_MASS = 0.1
CUBE_RGBA = [1, 0, 0, 1]
SPHERE_RGBA = [0.3, 1, 0, 1]

# Gripper constants
GRIPPER_OPEN_WIDTH = 0.07
GRIPPER_CLOSED_WIDTH = 0.007

# Grasp sequence constants
GRASP_APPROACH_HEIGHT_OFFSET = 0.1
GRASP_POSITION_OFFSET = 0.01
TRAY_DROP_HEIGHT_OFFSET = 0.15
TRAY_DROP_ORIENTATION = [np.pi, 0, 0]


@dataclass
class ObjectInfo:
    """Object information for simulation."""
    urdf_path: str
    position: List[float]
    orientation: List[float] = field(default_factory=lambda: [0, 0, 0])
    scaling: float = 1.0
    color: Optional[List[float]] = None  # RGBA color, e.g. [1, 0, 0, 1]
    mass: Optional[float] = None  # Mass in kg


class Sim:
    def __init__(
        self,
        urdf_path: Optional[str] = None,
        start_pos: List[float] = [0, 0, 0],
        start_orientation: List[float] = [0, 0, 0],  # Euler angles [roll, pitch, yaw]
    ):
        """
        Initializes the PyBullet simulation environment and loads the robot.

        Args:
            urdf_path (Optional[str]): Path to the robot's URDF file.
            start_pos (List[float]): Initial base position [x, y, z]. Defaults to [0, 0, 0].
            start_orientation (List[float]): Initial base orientation [roll, pitch, yaw] in radians. Defaults to [0, 0, 0].
        """
        self.physicsClient = self.setup_simulation()
        self.start_pos = start_pos
        self.start_orientation = start_orientation

        # Convert Euler angles to quaternion for PyBullet
        quaternion_orientation = pb.getQuaternionFromEuler(self.start_orientation)

        # Load the URDF
        self.robot_id = pb.loadURDF(
            urdf_path,
            basePosition=self.start_pos,
            baseOrientation=quaternion_orientation,
            useFixedBase=True,
        )
        self.num_joints = pb.getNumJoints(self.robot_id)
        self.lower_limits = []
        self.upper_limits = []
        self.joint_ranges = []
        self.rest_poses = [0.0] * self.num_joints  # Initialize with zeros

        for joint_index in range(self.num_joints):
            joint_info = pb.getJointInfo(self.robot_id, joint_index)
            joint_type = joint_info[2]

            # Get joint limits
            if joint_type == pb.JOINT_REVOLUTE:
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.lower_limits.append(lower_limit)
                self.upper_limits.append(upper_limit)
                self.joint_ranges.append(upper_limit - lower_limit)
            else:
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.lower_limits.append(lower_limit)
                self.upper_limits.append(upper_limit)
                self.joint_ranges.append(upper_limit - lower_limit)


    def __del__(self):
        """
        Ensures the PyBullet physics client is disconnected upon object garbage collection.
        """
        if hasattr(self, "physicsClient") and pb.isConnected(self.physicsClient):
            pb.disconnect(self.physicsClient)
            print("Disconnected from PyBullet physics server")

    def setup_simulation(self) -> int:
        """
        Sets up the PyBullet physics simulation environment.

        Connects to the physics server (GUI), sets gravity, loads the ground plane,
        and configures the simulation environment.

        Returns:
            int: The physics client ID assigned by PyBullet.

        Raises:
            ConnectionError: If connection to the PyBullet simulation fails.
        """
        physicsClient = pb.connect(pb.GUI)

        if physicsClient < 0:
            raise ConnectionError("Failed to connect to PyBullet simulation.")

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)
        pb.loadURDF("plane.urdf")
        return int(physicsClient)


class SimGrasp(Sim):
    def __init__(
        self,
        objects: List[ObjectInfo] = None,
        urdf_path: Optional[str] = None,
        start_pos: List[float] = [0, 0, 0],
        start_orientation: List[float] = [0, 0, 0],  # Euler angles [roll, pitch, yaw]
        frequency: int = 30,
    ):
        """
        Initializes the simulation environment specifically for grasping tasks.

        Inherits from Sim and adds object loading, camera setup, and grasping-specific parameters.

        Args:
            objects (List[ObjectInfo]): List of objects to load into the simulation.
            urdf_path (Optional[str]): Path to the robot's URDF file.
            start_pos (List[float]): Initial base position [x, y, z]. Defaults to [0, 0, 0].
            start_orientation (List[float]): Initial base orientation [roll, pitch, yaw] in radians. Defaults to [0, 0, 0].
            frequency (int): Simulation frequency in Hz for control loops. Defaults to 30.
        """
        super().__init__(urdf_path, start_pos, start_orientation)

        self.realsensed435_cam = cameras.RealSenseD435.CONFIG
        self._random = np.random.RandomState(None)
        self.frequency = frequency
        pb.changeDynamics(self.robot_id, 7, lateralFriction=6, spinningFriction=3)
        pb.changeDynamics(self.robot_id, 8, lateralFriction=6, spinningFriction=3)
        
        # Dictionary to store object IDs
        self.object_ids = {}
        
        # Load all objects
        for i, obj in enumerate(objects):
            obj_id = self.add_object(
                obj.urdf_path, 
                obj.position, 
                obj.orientation, 
                globalScaling=obj.scaling
            )
            
            # Set object properties if specified
            if obj.color is not None:
                pb.changeVisualShape(obj_id, -1, rgbaColor=obj.color)
            
            if obj.mass is not None:
                pb.changeDynamics(obj_id, -1, mass=obj.mass)
            
            # Store object ID with a key based on the object type
            obj_type = obj.urdf_path.split('/')[-1].split('.')[0]
            if obj_type in self.object_ids:
                # If we already have this type, add index
                self.object_ids[f"{obj_type}_{i}"] = obj_id
            else:
                self.object_ids[obj_type] = obj_id

    
    def step_simulation(self) -> None:
        """
        Steps the simulation.
        """
        pb.stepSimulation()
        time.sleep(1 / self.frequency)


    def add_object(self, urdf_path: str, pos=[0.3, 0.15, 0.0], orientation=[np.pi / 2, 0, 0], globalScaling: float = 1.0) -> int:
        """

        Loads the 'urdf_path' model at a specified position and orientation,
        and sets its dynamics properties.

        Args:
            pos (List[float], optional): The [x, y, z] position to place the object. Defaults to [0.3, 0.15, 0.0].
            orientation (List[float], optional): The [roll, pitch, yaw] orientation in radians. Defaults to [np.pi / 2, 0, 0].

        Returns:
            int: The unique body ID assigned to the loaded object by PyBullet.
        """
        orientation = pb.getQuaternionFromEuler(orientation)
        id = pb.loadURDF(urdf_path, pos, orientation, globalScaling=globalScaling)
        pb.changeDynamics(id, -1, lateralFriction=2, spinningFriction=1)
        return id

    def render_camera(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Renders color, depth, and segmentation images from the simulated RealSense D435 camera.

        Uses the camera configuration defined in `self.realsensed435_cam` to compute
        view and projection matrices, then captures the image using PyBullet's OpenGL renderer.
        Applies optional noise to color and depth images if configured.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - color (np.ndarray): The RGB color image (height, width, 3) as a NumPy array.
                - depth (np.ndarray): The depth image (height, width) as a NumPy array (values in meters).
                - segm (np.ndarray): The segmentation mask (height, width) as a NumPy array.
        """

        config = self.realsensed435_cam[0]
        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Convert color data to numpy array and reshape
        color = np.array(color, dtype=np.uint8).reshape(config["image_size"][0], config["image_size"][1], 4)
        rgb = color[:, :, :3]  # Remove alpha channel
        self.obs = {'image': rgb, 'depth': depth, 'seg': segm}

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def create_pointcloud(self, color: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Creates an Open3D PointCloud object from color and depth images.

        Uses camera intrinsic parameters defined in `self.realsensed435_cam`.
        The point cloud is transformed to align with the expected coordinate frame.

        Args:
            color (np.ndarray): The RGB color image (height, width, 3).
            depth (np.ndarray): The depth image (height, width), with values in meters.

        Returns:
            o3d.geometry.PointCloud: The generated Open3D point cloud object.
        """

        camera_config = self.realsensed435_cam[0]
        # Convert numpy arrays to Open3D images
        o3d_color = o3d.geometry.Image(np.ascontiguousarray(color))
        o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))

        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,
            depth_trunc=10000.0,
            convert_rgb_to_intensity=False,
        )

        # Get camera intrinsics
        intrinsics_matrix = camera_config["intrinsics"]
        img_height, img_width = camera_config["image_size"]
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=img_width,
            height=img_height,
            fx=intrinsics_matrix[0, 0],
            fy=intrinsics_matrix[1, 1],
            cx=intrinsics_matrix[0, 2],
            cy=intrinsics_matrix[1, 2],
        )

        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

        # Flip pointcloud 180 around camera's x axis
        pcd.transform(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        return pcd

    def robot_control(self, gripper_width: float, tip_target_pos_: List[float], tip_target_orientation: Optional[List[float]] = None) -> None:
        """
        Controls the robot arm to reach a target end-effector pose and gripper width.

        Calculates inverse kinematics (IK) to find the required joint angles.
        If `tip_target_orientation` is provided, it attempts to match both position and orientation.
        Otherwise, it only matches the target position.
        Sets the target joint angles (including gripper joints) using position control
        and steps the simulation.

        Args:
            gripper_width (float): The desired distance between the gripper fingers.
            tip_target_pos_ (List[float]): The target [x, y, z] position for the end-effector (link EEF_IDX).
            tip_target_orientation (Optional[List[float]], optional): The target orientation
                for the end-effector as a quaternion [qx, qy, qz, qw]. Defaults to None.
        """
        if tip_target_orientation:
            target_joint_angles = pb.calculateInverseKinematics(
                self.robot_id,
                EEF_IDX,
                targetPosition=tip_target_pos_,
                targetOrientation=tip_target_orientation,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.rest_poses,
                maxNumIterations=1000,
            )
        else:
            target_joint_angles = pb.calculateInverseKinematics(
                self.robot_id,
                EEF_IDX,
                targetPosition=tip_target_pos_,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.rest_poses,
                maxNumIterations=1000,
            )

        target_joint_angles = [x for x in target_joint_angles]
        target_joint_angles[7] = gripper_width / 2
        target_joint_angles[8] = -gripper_width / 2

        for i in range(30):
            for i in range(len(target_joint_angles)):
                pb.setJointMotorControl2(
                    self.robot_id,
                    i,
                    pb.POSITION_CONTROL,
                    target_joint_angles[i],
                    force=5 * 240.0,
                )
            pb.stepSimulation()
            time.sleep(1 / self.frequency)


    def transform_grasps_to_robot_frame(self, grasps_cam_frame: List[Grasp]) -> tuple[List[Grasp], np.ndarray, np.ndarray]:
        """
        Transforms a list of grasp poses from the camera's coordinate frame to the robot's base coordinate frame.

        Uses the `transform_cam_to_rob` function to perform the transformation for each grasp.

        Args:
            grasps_cam_frame (List[Grasp]): A list of Grasp objects defined in the camera's coordinate frame.

        Returns:
            tuple[List[Grasp], np.ndarray, np.ndarray]: A tuple containing:
                - transformed_grasps_robot_frame (List[Grasp]): The list of grasps transformed into the robot's frame.
                - all_transformed_rotations (np.ndarray): An array of the transformed rotation matrices (N, 3, 3).
                - all_transformed_translations (np.ndarray): An array of the transformed translation vectors (N, 3).
        """
        transformed_grasps_robot_frame = []
        all_transformed_rotations = []
        all_transformed_translations = []
        for grasp in grasps_cam_frame:
            rot_orig = np.array(grasp.rotation)
            trans_orig = np.array(grasp.translation)
            rot, trans = transform_cam_to_rob(rot_orig, trans_orig)
            transformed_grasps_robot_frame.append(Grasp(rotation=rot.tolist(), translation=trans.tolist()))
            all_transformed_rotations.append(rot)
            all_transformed_translations.append(trans)
        return transformed_grasps_robot_frame, np.array(all_transformed_rotations), np.array(all_transformed_translations)


    def execute_grasp_sequence(self, target_pose: List[float]) -> None:
        """
        Executes a predefined sequence of robot movements to perform a grasp.

        The sequence involves: moving above the target, moving to the target,
        closing the gripper, and lifting the object.

        Args:
            target_pose (List[float]): The target grasp pose, including position [x, y, z]
                and orientation quaternion [qx, qy, qz, qw].
        """
        # Move above the target pose with gripper open
        self.robot_control(
            GRIPPER_OPEN_WIDTH, [target_pose[0], target_pose[1], target_pose[2] + GRASP_APPROACH_HEIGHT_OFFSET], target_pose[3:]
        )
        # Move to the target grasp pose
        self.robot_control(
            GRIPPER_OPEN_WIDTH, [target_pose[0], target_pose[1], target_pose[2] - GRASP_POSITION_OFFSET], target_pose[3:]
        )
        # Close the gripper
        self.robot_control(
            GRIPPER_CLOSED_WIDTH, [target_pose[0], target_pose[1], target_pose[2] - GRASP_POSITION_OFFSET], target_pose[3:]
        )
        # Lift the object
        self.robot_control(
            GRIPPER_CLOSED_WIDTH, [target_pose[0], target_pose[1], target_pose[2] + GRASP_APPROACH_HEIGHT_OFFSET], target_pose[3:]
        )

    def drop_object_in_tray(self) -> None:
        """
        Moves the gripper over the tray, then opens it to drop the currently held object.
        """
        # Get tray position
        tray_id = self.tray_id if hasattr(self, 'tray_id') else self.object_ids.get('traybox', None)
        if tray_id is None:
            print("Warning: No tray found in simulation. Cannot drop object.")
            return
            
        tray_pos, _ = pb.getBasePositionAndOrientation(tray_id)
        # Define target position above the tray center
        drop_target_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + TRAY_DROP_HEIGHT_OFFSET] # Adjust height as needed
        drop_target_orientation = TRAY_DROP_ORIENTATION

        # Move above the tray with gripper closed
        self.robot_control(
            GRIPPER_CLOSED_WIDTH, drop_target_pos, drop_target_orientation
        )

        # Open the gripper while maintaining position and orientation
        self.robot_control(
            GRIPPER_OPEN_WIDTH, drop_target_pos, drop_target_orientation
        )

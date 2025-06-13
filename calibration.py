"""
This file is for calibrating your camera with respect to the robot, i.e. computing the transformation matrix between camera and robot frame.
This is a requirement for the grasp_example_so100.py file.

Instructions:
- Ensure SO100 is connected
- Ensure depth camera is connected and place it at desired position. We recommend an isometric view angle.
- Fill in the DEFAULT_PORT variable below with the port of your SO100. See https://huggingface.co/docs/lerobot/en/so101#setup-motors-video
- Put 12 small markers in reachable places for the robot and viewable by the depth camera.
- Run: python3 calibration.py --os LINUX or python3 calibration.py --os MAC
- If you haven't yet recorded robot poses enter 'y' and move the robot jaw tip to the center of each marker and press Enter. Repeat for all 12 points.
- Then a sequence of image windows will pop up. Click the marker centers in the same order as the robot poses. Make sure to close the image window after each click.
- The script will then compute the transformation matrix and save it to config/transform_mat.npy
- The script will also compute the scaling factor and save it to config/scaling_factor.npy
"""

import pybullet as pb
import pybullet_data
import time
import numpy as np
import threading
import argparse
from typing import List, Tuple
import os
from so100_client import SO100Client
from utils import get_3d_point_from_2d_coordinates
from capture_realsense_pointcloud import capture_pointcloud
from img_click import ImgClick
from compute_transform_mat import calibrate

# User TODO: Fill in the default port for your SO100 robot.
DEFAULT_PORT = "/dev/ttyACM0"
REAL_ROBOT_ROBOT_POSES_PATH = "config/robot_poses.npy"
URDF_PATH = "SO100/so100_calibration.urdf"
FREQUENCY = 100
DEFAULT_OS = "LINUX"
NUM_POINTS_USE = 12


def setup_simulation() -> int:
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


def get_joint_pose_in_world_coords(robot_id : int, joint_index : int) -> Tuple[List[float], List[float]]:
    """
    Get the position and orientation of a joint in world coordinates.

    Args:
        robot_id (int): The robot ID in PyBullet
        joint_index (int): The joint index

    Returns:
        Tuple[List[float], List[float]]: Position [x, y, z] and orientation [roll, pitch, yaw] in degrees
                                         of the joint in world coordinates
    """
    # Get the joint info which contains parent link index
    joint_info = pb.getJointInfo(robot_id, joint_index)
    parent_link_index = joint_info[16]

    # Get the joint's local position and orientation relative to parent link
    joint_parent_pos = joint_info[14]  # Position of joint in parent link frame
    joint_parent_orn = joint_info[15]  # Orientation of joint in parent link frame

    # If the joint has a parent link
    if parent_link_index != -1:
        # Get the parent link's state in world coordinates
        parent_link_state = pb.getLinkState(robot_id, parent_link_index)
        parent_link_pos = parent_link_state[0]  # World position
        parent_link_orn = parent_link_state[1]  # World orientation as quaternion

        # Convert the joint's local position and orientation to world coordinates
        # This transforms from parent link frame to world frame
        world_pos, world_orn = pb.multiplyTransforms(
            parent_link_pos, parent_link_orn, joint_parent_pos, joint_parent_orn
        )
    else:
        # If joint has no parent link, it's directly connected to base
        base_pos, base_orn = pb.getBasePositionAndOrientation(robot_id)
        world_pos, world_orn = pb.multiplyTransforms(
            base_pos, base_orn, joint_parent_pos, joint_parent_orn
        )
    # Convert quaternion to Euler angles in degrees
    euler_angles = pb.getEulerFromQuaternion(world_orn)
    euler_degrees = np.degrees(euler_angles).tolist()

    return world_pos, euler_degrees



class RobotPoseRecorder:
    def __init__(self, robot_id : int):
        """
        Initialize the RobotPoseRecorder state.
        Initializes the lock for appending to robot_poses and connects to the so100 leader client.

        Args:
            robot_id (int): The ID of the robot in PyBullet
        """
        self.current_pos = None  # Will hold the most recent position of the monitored joint
        self._robot_pose_lock = threading.Lock()
        self.robot_poses = []
        self.so100_client = SO100Client(port=DEFAULT_PORT, follower=False, force_calibration=True)
        self.robot_id = robot_id

    def _input_listener(self):
        """Background thread waiting for the user to hit Enter to record a pose.

        The thread blocks on standard input so the main simulation loop can run
        without interruption. When the user presses just the Enter key (empty
        string), the latest value of ``current_pos`` is appended to
        ``robot_poses``.  A lock is used to avoid race conditions with the main
        thread updating ``current_pos``.
        """
        while True:
            try:
                user_in = input()
            except EOFError:
                # Stream closed – terminate thread
                break

            if user_in == "":
                # User pressed Enter with no additional input => record pose
                with self._robot_pose_lock:
                    if self.current_pos is not None:
                        self.robot_poses.append(self.current_pos.copy())
                        print(f"[INFO] Pose recorded: {self.current_pos}")
            elif user_in.lower() in {"q", "quit", "exit"}:
                print("[INFO] Exit command received. Terminating simulation loop …")
                # Signal the simulation loop to stop by raising SystemExit in main
                import os, signal, sys
                os.kill(os.getpid(), signal.SIGINT)
                break


    def robot_pose_recorder(self) -> np.ndarray:
        """
        Record robot poses by connecting to the so100 leader client, reading the joint angles when 
        'Enter' is pressed, then going FK on the EEF link in pybullet to get the 3D position in robot frame.
        Repeats this until NUM_POINTS_USE poses are recorded.

        Returns:
            np.ndarray: The recorded robot poses.
        """

        # Start the background listener thread before entering simulation loop
        _input_thread = threading.Thread(target=self._input_listener, daemon=True)
        _input_thread.start()

        print("--------------------------------")
        print("Press Enter to record a pose...")
        print("--------------------------------")
        while len(self.robot_poses) < NUM_POINTS_USE:
            joint_angles = self.so100_client.read_joints()
            joint_angles = [0] + joint_angles
            for i in range(len(joint_angles)):
                pb.setJointMotorControl2(
                    self.robot_id,
                    i,
                    pb.POSITION_CONTROL,
                    joint_angles[i],
                    force=5 * 240.0,
                )
            pos, _ = get_joint_pose_in_world_coords(self.robot_id, 7)

            # Update shared position under lock to avoid race conditions
            with self._robot_pose_lock:
                self.current_pos = list(pos)  # Make sure it's a mutable list copy
            pb.stepSimulation()
            time.sleep(1 / FREQUENCY)

        self.robot_poses = np.array(self.robot_poses)
        return self.robot_poses


def camera_pose_recorder(os: str) -> np.ndarray:
    """
    Record camera poses by clicking on the center of the tapes on the image.
    The user must exit the viewer each time to continue to the next tape.

    Returns:
        np.ndarray: The recorded 3D positions in camera frame.
    """
    # User clicks on centers of tapes on image to get camera poses
    pcd = capture_pointcloud()
    camera_poses = np.zeros((NUM_POINTS_USE, 3))
    for i in range(NUM_POINTS_USE):
        img_click = ImgClick(np.asarray(pcd.colors), os=os)
        x, y = img_click.run()
        # Validate that user made a selection
        assert x is not None and y is not None, "No object clicked - please run again and click on an object"
        # Get the 3D point from the 2D click
        place_point = get_3d_point_from_2d_coordinates(pcd, x, y)
        if place_point[0] == 0. and place_point[1] == 0. and place_point[2] == 0.:
            print("Warning: captured pointcloud is incomplete. Is the lighting bad?")
            exit(0)
        camera_poses[i] = place_point
    return camera_poses


def main():
    """
    Main function to run the calibration. 
    - Loads the URDF
    - Optionally records robot poses
    - Records camera poses
    - Computes the transformation matrix between camera and robot frame
    - Saves results to /config directory
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robot calibration script')
    parser.add_argument('--os', type=str, required=True, choices=['LINUX', 'MAC'],
                       help='Operating system, options: LINUX, MAC')
    args = parser.parse_args()
        
    setup_simulation()

    if not os.path.exists("config"):
        os.makedirs("config")
    robot_id = pb.loadURDF(
        URDF_PATH,
        basePosition=[0,0,0],
        baseOrientation=pb.getQuaternionFromEuler([0,0,0]),
        useFixedBase=True,
    )

    if input("Do you want to record robot poses? (y/n): ") == "y":
        print("Recording robot poses...")
        robot_pose_recorder = RobotPoseRecorder(robot_id)
        robot_poses = robot_pose_recorder.robot_pose_recorder()
    else: 
        print("Loading robot poses...")
        robot_poses = np.load(REAL_ROBOT_ROBOT_POSES_PATH)
    print("Recording camera poses...")
    print("Clear camera view to tape.. (3s until capture)")
    time.sleep(3)
    camera_poses = camera_pose_recorder(args.os)
    print("Calibrating...")
    calibrate(camera_poses, robot_poses)


if __name__ == "__main__":
    main()
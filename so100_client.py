from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.common.teleoperators.utils import make_teleoperator_from_config
from typing import List, Tuple
import time
import numpy as np
import os
from pathlib import Path

# To account for calibration different between SO-100 URDF and physical SO-100
OFFSET = [3.5926674e-02,  1.7880281e+00, -1.5459236e+00, -6.0479313e-01,
  7.6940347e-04,  0]
CALIBRATION_PATH = "config/so100_calibration.json"


class SO100Client:
    def __init__(self, port: str, limits : List[Tuple[float, float]] = None, follower: bool = True, force_calibration: bool = False):
        """
        Initialize the SO-100 client.

        Args:
            port (str): The port to connect to the SO-100 via USB.
            limits (List[Tuple[float, float]]): The pairs of upper and lower limits of the joints.
        """
        self.follower = follower
        if self.follower:
            print("Using follower config")
            self.cfg = SO100FollowerConfig(port=port)
            self.cfg.use_degrees = True
            self.robot = make_robot_from_config(self.cfg)
        else:
            self.cfg = SO100LeaderConfig(port=port)
            self.cfg.use_degrees = True
            self.robot = make_teleoperator_from_config(self.cfg)
        self.robot.connect(calibrate=False)
        if not os.path.exists(CALIBRATION_PATH) or force_calibration:
            self.robot.calibrate()
            self.robot._save_calibration(Path(CALIBRATION_PATH))
        else:
            self.robot._load_calibration(Path(CALIBRATION_PATH))

        self.limits = limits
        print(f"Limits: {self.limits}")
    

    def write_joints(self, joints : List[float]):
        """Send joint command in radians to the robot.

        The sequence is:
            1. Subtract the calibration *OFFSET* provided at the top of the file
               so that `joints = 0` corresponds to the follower's neutral pose.
            2. Convert the resulting radian values to the [-100, 100] scale.
            3. Dispatch the command dictionary to `self.robot.send_action`.

        Args:
            joints (List[float]): List of joint angles in **radians**.
        """

        if len(joints) != len(OFFSET):
            raise ValueError(
                f"Expected {len(OFFSET)} joint values, got {len(joints)}: {joints}"
            )

        if self.limits is not None:
            for i in range(len(joints)):
                assert joints[i] >= self.limits[i][0] and joints[i] <= self.limits[i][1], f"Joint {i} is out of bounds: {joints[i]} for limits: {self.limits[i]}"

        # 1. Apply offset so that the policy can work in its own reference frame.
        offset_joints = [j - o for j, o in zip(joints, OFFSET)]

        offset_joints[0] = -offset_joints[0]


        normalised = np.degrees(offset_joints)

        # 2. Build the action dictionary.
        action = {
            "shoulder_pan.pos": normalised[0],
            "shoulder_lift.pos": normalised[1],
            "elbow_flex.pos": normalised[2],
            "wrist_flex.pos": normalised[3],
            "wrist_roll.pos": normalised[4],
            "gripper.pos": normalised[5],
        }

        self.robot.send_action(action)
        
    def read_joints(self) -> List[float]:
        """
        Read the joint angles from the robot in radians.

        Returns:
            List[float]: List of joint angles in radians.
        """
        if self.follower:
            observation = self.robot.get_observation()
        else:
            observation = self.robot.get_action()
        res = np.array([observation['shoulder_pan.pos'], 
                observation['shoulder_lift.pos'], 
                observation['elbow_flex.pos'], 
                observation['wrist_flex.pos'], 
                observation['wrist_roll.pos'], 
                observation['gripper.pos']])
        res = np.radians(res)
        offset_joints = [j + o for j, o in zip(res, OFFSET)]
        offset_joints[0] = -offset_joints[0]
        return offset_joints

    def interpolate_waypoint(self, waypoint1, waypoint2, steps=50, timestep=0.02):
        for i in range(steps):
            alpha = i / (steps-1)  # Goes from 0 to 1
            q = [(1-alpha)*w1 + alpha*w2 for w1, w2 in zip(waypoint1, waypoint2)]
            self.write_joints(q)
            time.sleep(timestep)
    
    def __del__(self):
        self.robot.disconnect()
    
if __name__ == "__main__":
    client = SO100Client(port="/dev/ttyACM0")
    action = {
            "shoulder_pan.pos": 0,
            "shoulder_lift.pos": 0,
            "elbow_flex.pos": 0,
            "wrist_flex.pos": 0,
            "wrist_roll.pos": 0,
            "gripper.pos": 0,
        }
        
    client.robot.send_action(action)
    time.sleep(1)
    print(client.robot.get_observation())
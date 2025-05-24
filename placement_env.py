import numpy as np
import pybullet as pb
import pybullet_data
import random
from sim import (
    SimGrasp,
    ObjectInfo,
    CUBE_SCALING,
)
from scipy.spatial.transform import Rotation as R


def get_urdf_bounding_box(urdf_path: str, scaling: float = 1.0) -> float:
    """
    Get the bounding box size of a URDF object.

    Args:
        urdf_path: Path to the URDF file
        scaling: Scaling factor applied to the object

    Returns:
        Approximate size (width) of the object
    """
    # Create temporary physics client to load the object
    temp_client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    try:
        # Load the object
        obj_id = pb.loadURDF(urdf_path, globalScaling=scaling)

        # Get the AABB (Axis-Aligned Bounding Box)
        aabb_min, aabb_max = pb.getAABB(obj_id)

        # Calculate the size in X direction (width)
        size_x = aabb_max[0] - aabb_min[0]

        pb.disconnect(temp_client)
        return size_x

    except Exception as e:
        print(
            f"Warning: Could not load URDF {urdf_path}, using default size. Error: {e}"
        )
        pb.disconnect(temp_client)
        return 0.05  # Default fallback size


def generate_random_block_positions(
    base_y=0.0, block_size=None, urdf_path="cube_small.urdf", scaling=CUBE_SCALING
):
    """
    Generate random positions for two blocks with one block space between them.

    Args:
        base_y: Base Y coordinate for the blocks
        block_size: Size of each block (auto-detected from URDF if None)
        urdf_path: Path to the URDF file to measure
        scaling: Scaling factor applied to the URDF

    Returns:
        Tuple of (left_block_pos, right_block_pos, target_pos)
    """
    # Auto-detect block size from URDF if not provided
    if block_size is None:
        block_size = get_urdf_bounding_box(urdf_path, scaling)
        print(f"Detected block size from URDF: {block_size:.4f} meters")
        # Generate random position and orientation for the left block
    left_x = random.uniform(0.2, 0.35)
    left_pos = [left_x, base_y, 0]

    # Generate same random orientation for both stationary blocks
    block_orientation = [
        random.uniform(-0.3, 0.3),  # Random roll
        random.uniform(-0.3, 0.3),  # Random pitch
        random.uniform(-np.pi, np.pi),  # Random yaw
    ]

    # Calculate right block position based on left block's X-axis direction
    # Convert Euler angles to rotation matrix to get X-axis direction

    rotation_matrix = R.from_euler("xyz", block_orientation).as_matrix()
    x_axis_direction = rotation_matrix[:, 0]  # First column is X-axis direction

    # Position right block one block size away along left block's X-axis
    displacement = x_axis_direction * block_size * 2
    right_pos = [
        left_pos[0] + displacement[0],
        left_pos[1] + displacement[1],
        left_pos[2] + displacement[2],
    ]

    # Target position is halfway between left and right blocks
    target_pos = [
        (left_pos[0] + right_pos[0]) / 2,
        (left_pos[1] + right_pos[1]) / 2,
        (left_pos[2] + right_pos[2]) / 2,
    ]

    return left_pos, right_pos, target_pos, block_orientation

def setup_placement_scenario():
        """
        Sets up a placement scenario with three blocks - two stationary blocks and one target block.
        
        Returns:
            SimGrasp: Configured simulation environment with the blocks placed
        """
        # Generate random positions for the blocks (auto-detect size from URDF)
        left_block_pos, right_block_pos, target_block_pos, block_orientation = (
            generate_random_block_positions(
                urdf_path="cube_small.urdf", scaling=CUBE_SCALING
            )
        )

        print(f"Left block position: {left_block_pos}")
        print(f"Right block position: {right_block_pos}")
        print(f"Target block position (for robot to place): {target_block_pos}")

        # Create different colored blocks for visual distinction
        left_block_color = [1, 0, 0, 1]  # Red
        right_block_color = [0, 0, 1, 1]  # Blue
        target_block_color = [0, 1, 0, 1]  # Green (the one robot should move)

        return SimGrasp(
            urdf_path="piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf",
            frequency=30,
            objects=[
                # Left LEGO block (same random orientation as right)
                ObjectInfo(
                    urdf_path="cube_small.urdf",
                    position=left_block_pos,
                    orientation=block_orientation,
                    scaling=CUBE_SCALING,
                    color=left_block_color,
                ),
                # Right LEGO block (same random orientation as left)
                ObjectInfo(
                    urdf_path="cube_small.urdf",
                    position=right_block_pos,
                    orientation=block_orientation,
                    scaling=CUBE_SCALING,
                    color=right_block_color,
                ),
                # Target block that robot should move to the middle (keep standard orientation for easier grasping)
                ObjectInfo(
                    urdf_path="cube_small.urdf",
                    position=[
                        target_block_pos[0],
                        target_block_pos[1] - 0.1,
                        target_block_pos[2],
                    ],  # Place it slightly away
                    orientation=[
                        random.uniform(-0.3, 0.3),  # Random roll
                        random.uniform(-0.3, 0.3),  # Random pitch
                        random.uniform(-np.pi, np.pi),  # Random yaw
                    ],
                    scaling=CUBE_SCALING,
                    color=target_block_color,
                ),
            ],
        )

if __name__ == "__main__":
    sim = setup_placement_scenario()

    print("Scenario setup complete!")
    print("- Red block (left)")
    print("- Blue block (right)")
    print("- Green block (to be moved to the middle)")
    print(
        "Robot's task: Move the green block to fill the gap between red and blue blocks"
    )
    # Keep simulation running
    while True:
        sim.step_simulation()
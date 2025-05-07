"""
Client script to demonstrate the grasp prediction pipeline.

Steps:
1. Initialize simulation environment.
2. Render camera view and generate initial point cloud.
3. Call Point Cloud Cropping service.
4. Call Grasp Prediction service on the cropped point cloud.
5. Transform predicted grasps from camera frame to robot frame.
6. Call Grasp Filtering service to get valid grasps.
7. Visualize the valid grasps.
8. Execute the first valid grasp in the simulation.
"""


import numpy as np
from scipy.spatial.transform import Rotation as R

from client import GeneralBionixClient
from vis import ImgClick
from sim import SimGrasp


# User TODO
API_KEY = "" # Use your API key here
OS = "LINUX" # "MAC" or "LINUX"

# Feel free to change these positions, but note that some positions are unreachable by the robot with IK.
DUCK_POS = [0.2, 0.05, 0.]
CUBE_POS = [0.3, 0., 0.025]
SPHERE_POS = [0.35, 0.1, 0.025]

FREQUENCY = 30
URDF_PATH = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"
DOWN_SAMPLE = 4 # Don't change this




def main():
    """Main execution function for the grasp prediction pipeline."""
    # Initialize the simulation environment
    env = SimGrasp(urdf_path=URDF_PATH, frequency=FREQUENCY, duck_pos=DUCK_POS, cube_pos=CUBE_POS, sphere_pos=SPHERE_POS)
    client = GeneralBionixClient(api_key=API_KEY)

    # Render camera image and create initial point cloud
    color, depth, _ = env.render_camera()
    pcd = env.create_pointcloud(color, depth)

    # Get the x,y coordinates of the object the user clicks on
    print("<Click on the object you want to grasp then close the image window>")
    img_click = ImgClick(np.asarray(pcd.colors), os=OS)
    x, y = img_click.run()

    assert x is not None and y is not None, "No object clicked"

    # Downsample for faster processing
    pcd = pcd.uniform_down_sample(DOWN_SAMPLE) # Downsample for faster processing

    print("Requesting Point Cloud Cropping service...")
    # 1. Crop Point Cloud via external service
    cropped_pcd_data = client.crop_point_cloud(pcd, int(x/DOWN_SAMPLE), y)

    print("Requesting Grasp Prediction service...")
    # 2. Predict Grasps via external service (grasps are in camera frame)
    grasps_response = client.predict_grasps(cropped_pcd_data)
    predicted_grasps_cam_frame = grasps_response.grasps

    # 3. Transform predicted grasps from camera frame to robot base frame
    predicted_grasps_robot_frame, all_transformed_rotations, all_transformed_translations = \
        env.transform_grasps_to_robot_frame(predicted_grasps_cam_frame)

    print("Requesting Grasp Filtering service...")
    # 4. Filter Grasps for reachability via external service
    filter_response = client.filter_grasps(predicted_grasps_robot_frame)
    valid_grasp_idxs = filter_response.valid_grasp_idxs

    if not valid_grasp_idxs:
        print("No valid grasps found after filtering.")
        return

    # Select the valid grasps based on indices returned by the filtering service (for execution)
    valid_rotations = all_transformed_rotations[np.array(valid_grasp_idxs)]
    valid_translations = all_transformed_translations[np.array(valid_grasp_idxs)]

    # Select the first valid grasp for execution
    # Convert rotation matrix to Euler angles for robot control
    pose_orientation_euler = R.from_matrix(valid_rotations[0]).as_euler('xyz', degrees=False).tolist()
    pose_translation = valid_translations[0].tolist()
    target_pose = pose_translation + pose_orientation_euler

    # 5. Execute the grasp sequence in simulation
    env.execute_grasp_sequence(target_pose)

    env.drop_object_in_tray()

    while True:
        env.step_simulation()


if __name__ == "__main__":
    main()

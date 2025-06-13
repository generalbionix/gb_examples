"""
Client script to demonstrate the grasp prediction pipeline.

This script demonstrates a complete end-to-end grasp prediction workflow:
1. Initialize simulation environment with objects to grasp
2. Render camera view and generate initial point cloud
3. Allow user to select target object via mouse click
4. Call Point Cloud Cropping service to isolate the target object
5. Call Grasp Prediction service on the cropped point cloud
6. Transform predicted grasps from camera frame to robot frame
7. Call Grasp Filtering service to get kinematically valid grasps
8. Visualize the valid grasps in 3D
9. Execute the best valid grasp in the simulation
10. Drop the grasped object into a tray

The pipeline integrates computer vision, machine learning, and robotics
to demonstrate autonomous object manipulation.
"""

import numpy as np
from typing import List
from vis_grasps import vis_grasps_meshcat
from transform import transform_pcd_cam_to_rob, grasp_service_to_robot_format, make_robot_config
import open3d as o3d
from client import GeneralBionixClient, PointCloudData, Grasp
from img_click import ImgClick
from vis_grasps import launch_visualizer
from utils import get_3d_point_from_2d_coordinates
from so100_client import SO100Client
from sim import SimGrasp
from capture_realsense_pointcloud import capture_pointcloud


# User TODO
OS = "LINUX" # "MAC" or "LINUX"
DEFAULT_PORT = "/dev/ttyACM0" # Get this from these instructions: https://huggingface.co/docs/lerobot/en/so101#1-find-the-usb-ports-associated-with-each-arm
API_KEY = "" # Currently our service doesn't require an API key so leave this empty!

SIMULATION_OBJECTS = [] # Empty sim env
REAL_ROBOT = True
FREQUENCY = 30
URDF_PATH = "SO100/so100.urdf"


def main():
    """
    Main execution function for the grasp prediction pipeline.
    
    This function orchestrates the complete workflow from scene setup
    to grasp execution, integrating multiple services and components.
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Initialize Simulation Environment
    # -------------------------------------------------------------------------
    print("Initializing simulation environment...")
    env = SimGrasp(urdf_path=URDF_PATH, frequency=FREQUENCY, objects=SIMULATION_OBJECTS)
    
    # Initialize API client for grasp prediction services
    client = GeneralBionixClient(api_key=API_KEY)
    
    # Launch 3D visualizer for displaying grasps
    vis = launch_visualizer()

    # SO-100 hardware interface
    so100_client = SO100Client(port=DEFAULT_PORT)

    robot_config = make_robot_config("so100")

    while True:

        # -------------------------------------------------------------------------
        # Step 2: Capture Scene and Generate Point Cloud
        # -------------------------------------------------------------------------
        print("Capturing camera view and generating point cloud...")
        pcd = capture_pointcloud()

        # -------------------------------------------------------------------------
        # Step 3: Interactive Object Selection
        # -------------------------------------------------------------------------
        print("\n" + "="*60)
        print("OBJECT SELECTION")
        print("="*60)
        print("Click on the object you want to grasp, then close the image window")
        
        # Launch interactive image viewer for user to select target object
        img_click = ImgClick(np.asarray(pcd.colors), os=OS)
        x, y = img_click.run()

        # Validate that user made a selection
        assert x is not None and y is not None, "No object clicked - please run again and click on an object"

        # -------------------------------------------------------------------------
        # Step 5: Point Cloud Cropping Service
        # -------------------------------------------------------------------------
        print("Requesting Point Cloud Cropping service...")
        
        # Call external service to crop point cloud around user-selected object
        # Adjusts click coordinates for downsampled point cloud
        cropped_pcd_data = client.crop_point_cloud(pcd, int(x), y)

        # Convert service response back to Open3D point cloud format
        cropped_pcd_cam_frame = o3d.geometry.PointCloud()
        cropped_pcd_cam_frame.points = o3d.utility.Vector3dVector(np.array(cropped_pcd_data.points))
        cropped_pcd_cam_frame.colors = o3d.utility.Vector3dVector(np.array(cropped_pcd_data.colors))


        # Upsample cropped point cloud back to original resolution
        # This ensures we maintain detail while benefiting from faster cropping
        # -------------------------------------------------------------------------
        # Step 6: Coordinate Frame Transformations
        # -------------------------------------------------------------------------
        print("Transforming point clouds to robot coordinate frame...")
        # Transform cropped point cloud from camera frame to robot base frame
        # This is necessary because grasp planning works in robot coordinates
        cropped_pcd_robot_frame = transform_pcd_cam_to_rob(cropped_pcd_cam_frame, real_robot=REAL_ROBOT)
        
        # Also transform full scene point cloud for visualization
        pcd_robot_frame = transform_pcd_cam_to_rob(pcd, real_robot=REAL_ROBOT)
        
        # Prepare cropped point cloud data for grasp prediction service
        cropped_pcd_data_robot_frame = PointCloudData(
            points=np.array(cropped_pcd_robot_frame.points).tolist(),
            colors=np.array(cropped_pcd_robot_frame.colors).tolist()
        )

        # -------------------------------------------------------------------------
        # Step 7: Grasp Prediction Service
        # -------------------------------------------------------------------------
        print("Requesting Grasp Prediction service...")
        
        # Call external ML service to predict grasp poses on the cropped object
        # Returns 6DOF grasp poses (position + orientation) in robot frame
        grasps_response = client.predict_grasps(cropped_pcd_data_robot_frame)
        predicted_grasps_robot_frame = grasps_response.grasps

        print(f"Generated {len(predicted_grasps_robot_frame)} potential grasp candidates")

        # -------------------------------------------------------------------------
        # Step 8: Grasp Filtering Service
        # -------------------------------------------------------------------------
        print("Requesting Grasp Filtering service...")
        print(f"predicted_grasps_robot_frame: {predicted_grasps_robot_frame[0]}")

        predicted_grasps_robot_frame_so100 = grasp_service_to_robot_format(predicted_grasps_robot_frame, robot_config)
        
        # Call external service to filter grasps for kinematic reachability
        # This ensures the robot can actually achieve the predicted grasp poses
        filter_response = client.filter_grasps(predicted_grasps_robot_frame_so100, robot_name="so100")
        valid_grasp_idxs = filter_response.valid_grasp_idxs
        valid_grasp_joint_angles = filter_response.valid_grasp_joint_angles

        # vis_grasps_meshcat(vis, predicted_grasps_robot_frame, pcd_robot_frame)

        # Check if any valid grasps were found
        if not valid_grasp_idxs:
            print("No valid grasps found after filtering.")
            return

        # Extract valid grasps from the full set of predictions
        valid_grasps: List[Grasp] = [predicted_grasps_robot_frame[i] for i in valid_grasp_idxs]
        
        print(f"✅ Found {len(valid_grasps)} kinematically valid grasps")

        # -------------------------------------------------------------------------
        # Step 9: Grasp Selection and Visualization
        # -------------------------------------------------------------------------
        
        # Select the first valid grasp for execution
        # In a real application, you might rank grasps by quality metrics
        chosen_grasp_idx = 0
        while True:
            chosen_grasp = valid_grasps[chosen_grasp_idx]

            vis_grasps_meshcat(vis, [valid_grasps[chosen_grasp_idx]], pcd_robot_frame, real_robot=REAL_ROBOT)

            env.add_pointcloud(cropped_pcd_robot_frame)

            # Visualize all valid grasps in 3D viewer
            print("Launching 3D visualization of valid grasps...")
            print("Check the MeshCat visualizer to see the grasp poses")

            # Add visual debug marker at chosen grasp location in simulation
            env.add_debug_point(chosen_grasp.translation)

            # -------------------------------------------------------------------------
            # Step 10: Grasp Execution In Simulation
            # -------------------------------------------------------------------------

            print("Executing the selected grasp...")

            # ---- so100_waypoint<n> is the joint angles for the nth waypoint on the physical robot ----
            # ---- All joint units in radians ----

            # Get joint angles for the chosen grasp from filtering service
            grasp_joint_angles = valid_grasp_joint_angles[chosen_grasp_idx]

            # Execute the grasp in simulation
            grasp_waypoints = env.so100_grasp_sequence(grasp_joint_angles, so100_client)

            # -------------------------------------------------------------------------
            # Step 11: Grasp Execution On Hadware
            # -------------------------------------------------------------------------
            user_input = input("Execute on so100? (y/n/c) Press c to continue to next grasp.")
            if user_input == "y":
                for i in range(len(grasp_waypoints)-1):
                    so100_client.interpolate_waypoint(grasp_waypoints[i], grasp_waypoints[i+1])
                break
            elif user_input == "c":
                chosen_grasp_idx += 1
                continue
            else: 
                exit(0)

        # -------------------------------------------------------------------------
        # Step 12: Object Placing
        # -------------------------------------------------------------------------
        print("Capturing new pointcloud...")
        pcd = capture_pointcloud()

        print("="*60)
        print("Click on the location you want to drop, then close the image window")
        
        # Launch interactive image viewer for user to select target object
        img_click = ImgClick(np.asarray(pcd.colors), os=OS)
        x, y = img_click.run()

        # Validate that user made a selection
        assert x is not None and y is not None, "No object clicked - please run again and click on an object"

        # Get the 3D point from the 2D click
        place_point = get_3d_point_from_2d_coordinates(pcd_robot_frame, x, y)

        # Robot IK target on the gripper link is 0.22m above the click point
        place_point[2] += 0.22

        # Visualize the place point in pybullet
        env.add_debug_point(place_point)

        place_waypoints = env.so100_place_sequence(place_point, so100_client)

        # -------------------------------------------------------------------------
        # Step 13: Object Placing On Hardware
        # -------------------------------------------------------------------------
        
        print("Executing placement on so100...")
        for i in range(len(place_waypoints)-1):
            so100_client.interpolate_waypoint(place_waypoints[i], place_waypoints[i+1])
    

    while True:
        env.step_simulation()


if __name__ == "__main__":
    main()

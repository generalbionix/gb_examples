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
from scipy.spatial.transform import Rotation as R
from vis_grasps import vis_grasps_meshcat
from transform import transform_pcd_cam_to_rob
import open3d as o3d
from client import PointCloudData, Grasp
from caching import CachedGeneralBionixClient
from img_click import ImgClick
from vis_grasps import launch_visualizer
from utils import get_3d_point_from_2d_coordinates
from sim import (
    SimGrasp, 
    ObjectInfo, 
    CUBE_ORIENTATION, 
    CUBE_SCALING, 
    SPHERE_ORIENTATION, 
    SPHERE_SCALING, 
    TRAY_ORIENTATION, 
    TRAY_SCALING, 
    TRAY_POS, 
    CUBE_RGBA, 
    SPHERE_RGBA, 
    SPHERE_MASS,
    DUCK_ORIENTATION,
    DUCK_SCALING,
)

REAL_ROBOT = False

# User TODO
API_KEY = "" # Use your API key here
OS = "LINUX" # "MAC" or "LINUX"

# Cache configuration - modify as needed
ENABLE_CACHE = False  # Set to False to disable caching
CACHE_VERBOSE = 0    # 0=silent, 1=normal, 2=verbose

# Define simulation objects
SIMULATION_OBJECTS = [
    ObjectInfo(
        urdf_path="cube_small.urdf",
        position=[0.35, 0.0, 0.025],
        orientation=CUBE_ORIENTATION,
        scaling=CUBE_SCALING,
        color=CUBE_RGBA
    ),
    ObjectInfo(
        urdf_path="cube_small.urdf",
        position=[0.25, 0., 0.025],
        orientation=CUBE_ORIENTATION,
        scaling=CUBE_SCALING,
        color=CUBE_RGBA
    ),
    ObjectInfo(
        urdf_path="tray/traybox.urdf",
        position=TRAY_POS,
        orientation=TRAY_ORIENTATION,
        scaling=TRAY_SCALING
    ),
    ObjectInfo(
        urdf_path="sphere2.urdf",
        position=[0.2, 0.1, 0.025],
        orientation=SPHERE_ORIENTATION,
        scaling=SPHERE_SCALING,
        color=SPHERE_RGBA,
        mass=SPHERE_MASS
    ),
    ObjectInfo(
        urdf_path="duck_vhacd.urdf",
        position=[0.3, 0.05, 0.],
        orientation=DUCK_ORIENTATION,
        scaling=DUCK_SCALING
    )
]

FREQUENCY = 30
URDF_PATH = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"




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
    
    # Initialize cached API client for grasp prediction services
    client = CachedGeneralBionixClient(
        api_key=API_KEY,
        enable_cache=ENABLE_CACHE,
        verbose=CACHE_VERBOSE
    )
    
    # Display cache statistics
    stats = client.cache_stats()
    if stats["cache_enabled"]:
        print(f"📊 Cache stats: {stats['total_files']} files, {stats['total_size_mb']} MB")
    
    # Launch 3D visualizer for displaying grasps
    vis = launch_visualizer()

    # -------------------------------------------------------------------------
    # Step 2: Capture Scene and Generate Point Cloud
    # -------------------------------------------------------------------------
    print("Capturing camera view and generating point cloud...")
    # Render RGB-D image from robot's camera perspective
    color, depth, _ = env.render_camera()
    
    # Convert RGB-D image to 3D point cloud in camera coordinate frame
    pcd = env.create_pointcloud(color, depth)

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
    
    # Call external service to filter grasps for kinematic reachability
    # This ensures the robot can actually achieve the predicted grasp poses
    filter_response = client.filter_grasps(predicted_grasps_robot_frame, robot_name="piper")
    valid_grasp_idxs = filter_response.valid_grasp_idxs
    valid_grasp_joint_angles = filter_response.valid_grasp_joint_angles

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
    chosen_grasp = valid_grasps[chosen_grasp_idx]
    
    print(f"Selected grasp {chosen_grasp_idx + 1} out of {len(valid_grasps)} valid options")
    print(f"Grasp position: [{chosen_grasp.translation[0]:.3f}, {chosen_grasp.translation[1]:.3f}, {chosen_grasp.translation[2]:.3f}]")

    # Visualize all valid grasps in 3D viewer
    print("Launching 3D visualization of valid grasps...")
    print("Check the MeshCat visualizer to see the grasp poses")
    vis_grasps_meshcat(vis, valid_grasps, pcd_robot_frame, real_robot=REAL_ROBOT)

    # Add visual debug marker at chosen grasp location in simulation
    env.add_debug_point(chosen_grasp.translation)

    # -------------------------------------------------------------------------
    # Step 10: Grasp Execution
    # -------------------------------------------------------------------------
    print("Executing the selected grasp...")
    
    # Get joint angles for the chosen grasp from filtering service
    grasp_joint_angles = valid_grasp_joint_angles[chosen_grasp_idx]
    
    # Execute the grasp in simulation
    print("Moving robot to grasp pose and closing gripper...")
    env.grasp(grasp_joint_angles)
    
    # -------------------------------------------------------------------------
    # Step 11: Capture Scene and Generate Point Cloud
    # -------------------------------------------------------------------------
    print("Capturing camera view and generating point cloud...")
    # Render RGB-D image from robot's camera perspective
    color, depth, _ = env.render_camera()
    
    # Convert RGB-D image to 3D point cloud in camera coordinate frame
    pcd = env.create_pointcloud(color, depth)

    # -------------------------------------------------------------------------
    # Step 12: Interactive Object Selection
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("OBJECT SELECTION")
    print("="*60)
    print("Click on the object you want to drop, then close the image window")
    
    # Launch interactive image viewer for user to select target object
    img_click = ImgClick(np.asarray(pcd.colors), os=OS)
    x, y = img_click.run()

    # Validate that user made a selection
    assert x is not None and y is not None, "No object clicked - please run again and click on an object"

    # Get the 3D point from the 2D click
    place_point = get_3d_point_from_2d_coordinates(pcd_robot_frame, x, y)

    # -------------------------------------------------------------------------
    # Step 13: Place the object
    # -------------------------------------------------------------------------
    print("Placing the object...")
    
    # Place the object
    env.place(place_point)
    while True:
        env.step_simulation()


if __name__ == "__main__":
    main()

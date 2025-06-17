"""
Demonstrates an object identification and grasping pipeline using a simulated robot.

The script continuously performs the following cycle:
1.  Captures a view from the simulation and generates a point cloud.
2.  Segments objects in the view, creating a visual prompt.
3.  Uses a multimodal model (GPT-4o) with a text query to identify target objects.
4.  If no targets are found, the script exits.
5.  Selects a target and calculates its center.
6.  Crops the point cloud around the target.
7.  Predicts grasp poses for the cropped point cloud.
8.  Transforms grasps to the robot's frame and filters for reachability.
9.  If no valid grasps, restarts the loop.
10. Executes a random valid grasp, drops the object, and advances the simulation.
"""


import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from typing import List
from client import PointCloudData, Grasp
from caching import CachedGeneralBionixClient
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
    SPHERE_MASS
)
from utils import compute_mask_center_of_mass

from visual_prompt.visual_prompt import VisualPrompterGrounding
from visual_prompt.utils import display_image
from vis_grasps import launch_visualizer
from utils import downsample_pcd, upsample_pcd
from vis_grasps import vis_grasps_meshcat
from transform import transform_pcd_cam_to_rob

REAL_ROBOT = False
# GPT-4o prompt
USER_QUERY = "Identify the red cubes. If no red cubes are available then return an empty list."

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
        position=[0.3, 0.0, 0.025],
        orientation=CUBE_ORIENTATION,
        scaling=CUBE_SCALING,
        color=CUBE_RGBA
    ),
    ObjectInfo(
        urdf_path="cube_small.urdf",
        position=[0.23, 0.05, 0.025],
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
        position=[0.35, 0.1, 0.025],
        orientation=SPHERE_ORIENTATION,
        scaling=SPHERE_SCALING,
        color=SPHERE_RGBA,
        mass=SPHERE_MASS
    )
]


FREQUENCY = 30
URDF_PATH = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"
CONFIG_PATH = 'visual_prompt/config/visual_prompt_config.yaml'





def main():
    """Main execution function for the grasp prediction pipeline."""
    # Initialize the simulation environment
    env = SimGrasp(urdf_path=URDF_PATH, frequency=FREQUENCY, objects=SIMULATION_OBJECTS)
    client = CachedGeneralBionixClient(
        api_key=API_KEY,
        enable_cache=ENABLE_CACHE,
        verbose=CACHE_VERBOSE
    )
    
    # Display cache statistics
    stats = client.cache_stats()
    if stats["cache_enabled"]:
        print(f"📊 Cache stats: {stats['total_files']} files, {stats['total_size_mb']} MB")
    
    vis = launch_visualizer()
    grounder = VisualPrompterGrounding(CONFIG_PATH, debug=True)

    while True:
        # Render camera image and generate initial point cloud from the simulation.
        color, depth, _ = env.render_camera()
        pcd = env.create_pointcloud(color, depth)
        # Prepare visual prompt: segment objects, create masks, and mark them on the image.
        image, seg = env.obs['image'], env.obs['seg']
        seg = np.array(seg).reshape(image.shape[:-1])
        obj_ids = np.unique(seg)[1:]
        all_masks = np.stack([seg == objID for objID in obj_ids])
        marker_data = {'masks': all_masks, 'labels': obj_ids}
        visual_prompt, _ = grounder.prepare_image_prompt(image.copy(), marker_data)
        marked_image_grounding = visual_prompt[-1]
        print("Displaying the visual prompt sent to GPT-4o...")
        display_image(marked_image_grounding, (6,6))
        print("Calling GPT-4o...")
        # Identify target objects using GPT-4o with the text query and visual prompt.
        _, _, target_ids = grounder.request(text_query=USER_QUERY,image=image.copy(),data=marker_data)
        if len(target_ids) == 0:
            print("No target objects identified by GPT-4o. Exiting.")
            exit(0)
        
        # Randomly select one of the identified target objects.
        target_idx = len(target_ids) - 1
        selected_target_id = target_ids[target_idx]
        print(f"Picked obj {selected_target_id}")
        # Compute the 2D center of mass of the selected target object's mask.
        center_x, center_y = compute_mask_center_of_mass(marker_data["masks"][marker_data["labels"].tolist().index(selected_target_id)])
        assert center_x is not None and center_y is not None, "No object clicked"
        # Downsample the point cloud for faster processing.
        print("Requesting Point Cloud Cropping service...")
        # Crop the point cloud around the target object using its 2D center.
        cropped_pcd_data = client.crop_point_cloud(pcd, int(center_x), int(center_y))

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
        
        # Transport grasped object to tray
        print("Transporting object to tray...")
        env.drop_object_in_tray()


if __name__ == "__main__":
    main()

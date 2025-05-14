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

from client import GeneralBionixClient
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




# GPT-4o prompt
USER_QUERY = "Identify the red cubes. If no red cubes are available then return an empty list."

# User TODO
API_KEY = "" # Use your API key here
OS = "LINUX" # "MAC" or "LINUX"


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
DOWN_SAMPLE = 4 # Don't change this
URDF_PATH = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"
CONFIG_PATH = 'visual_prompt/config/visual_prompt_config.yaml'





def main():
    """Main execution function for the grasp prediction pipeline."""
    # Initialize the simulation environment
    env = SimGrasp(urdf_path=URDF_PATH, frequency=FREQUENCY, objects=SIMULATION_OBJECTS)
    client = GeneralBionixClient(api_key=API_KEY)
    grounder = VisualPrompterGrounding(CONFIG_PATH, debug=True)

    while True:
      # Render camera image and generate initial point cloud from the simulation.
      color, depth, _ = env.render_camera()
      pcd = env.create_pointcloud(color, depth)

      # Prepare visual prompt: segment objects, create masks, and mark them on the image.
      image, seg = env.obs['image'], env.obs['seg']
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
      pcd = pcd.uniform_down_sample(DOWN_SAMPLE)

      print("Requesting Point Cloud Cropping service...")
      # Crop the point cloud around the target object using its 2D center.
      cropped_pcd_data = client.crop_point_cloud(pcd, int(center_x/DOWN_SAMPLE), int(center_y))

      print("Requesting Grasp Prediction service...")
      # Predict grasp poses on the cropped point cloud (grasps are in camera frame).
      grasps_response = client.predict_grasps(cropped_pcd_data)
      predicted_grasps_cam_frame = grasps_response.grasps

      # Transform predicted grasps from camera frame to the robot's base frame.
      predicted_grasps_robot_frame, all_transformed_rotations, all_transformed_translations = \
          env.transform_grasps_to_robot_frame(predicted_grasps_cam_frame)

      print("Requesting Grasp Filtering service...")
      # Filter grasps to keep only those reachable by the robot.
      filter_response = client.filter_grasps(predicted_grasps_robot_frame)
      valid_grasp_idxs = filter_response.valid_grasp_idxs

      if not valid_grasp_idxs:
          print("No valid grasps found after filtering. Restarting loop.")
          continue

      # Select valid grasps based on the indices returned by the filtering service.
      valid_rotations = all_transformed_rotations[np.array(valid_grasp_idxs)]
      valid_translations = all_transformed_translations[np.array(valid_grasp_idxs)]

      # Randomly select one valid grasp for execution.
      idx = np.random.randint(len(valid_rotations))
      # Convert rotation matrix to Euler angles for robot control.
      pose_orientation_euler = R.from_matrix(valid_rotations[idx]).as_euler('xyz', degrees=False).tolist()
      pose_translation = valid_translations[idx].tolist()
      target_pose = pose_translation + pose_orientation_euler

      # Execute the grasp sequence in the simulation.
      env.execute_grasp_sequence(target_pose)

      # Simulate dropping the grasped object into a tray.
      env.drop_object_in_tray()

      # Advance the simulation for a few steps.
      for _ in range(10):
          env.step_simulation()


if __name__ == "__main__":
    main()

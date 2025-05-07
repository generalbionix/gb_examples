# General Bionix Grasping Example

This repository demonstrates a pipeline for predicting and executing robot grasps using a PyBullet simulation environment and the General Bionix API for point cloud processing and grasp planning.


## Running the Example

**Prerequisites:**


```bash
conda create -n grasp_example python=3.10
conda activate grasp_example
pip install -r requirements.txt
```

**Add your API key and specify your OS ("MAC" or "LINUX") in the `API_KEY` and `OS` variables at the top of `grasp_example.py`.**

**Running the Example:**

```bash
python grasp_example.py
```
**After running the command, a PyBullet simulation window will pop up, and an image from the camera's perspective will be displayed. Click on an object in this image (except the tray which is fixed). The service will return grasp predictions. Close the grasp viewer to watch the robot grasp the object and place it in the tray!**

The object positions can be changed in `grasp_example.py` but note that our grasping model predicts grasp orientations relative to the camera position, so it may be hard to find IK solutions when moving the objects.

## Overview

The core components are:

1.  **`sim.py`**: A PyBullet-based simulation environment (`SimGrasp`) that loads a robot, objects, simulates a camera, and provides robot control functionalities.
2.  **`client.py`**: A client (`GeneralBionixClient`) to interact with external microservices for:
    *   Cropping point clouds.
    *   Predicting potential grasps from point clouds.
    *   Filtering grasps based on reachability.
3.  **`grasp_example.py`**: An example script orchestrating the entire pipeline: simulation setup, data acquisition, service calls, visualization, and grasp execution.
4.  **`vis.py`**: Utilities for visualizing point clouds and grasps, including user interaction for selecting regions.
5.  **`transform.py`**: Functions for coordinate transformations (e.g., camera frame to robot frame).
6.  **`cameras.py`**: Configuration details for simulated cameras.


## Example Workflow (`grasp_example.py`)

The `grasp_example.py` script demonstrates the end-to-end process:

1.  **Initialization**: Sets up the `SimGrasp` simulation environment and the `GeneralBionixClient`.
2.  **Rendering & Point Cloud**: Renders the scene from the simulated camera and generates an initial Open3D point cloud.
3.  **User Interaction**: Displays the RGB image and uses `vis.ImgClick` to capture the (x, y) pixel coordinates clicked by the user, indicating the region of interest.
4.  **Downsampling**: Reduces the point cloud density for faster processing.
5.  **Crop Point Cloud**: Calls `client.crop_point_cloud` using the clicked coordinates to get a focused point cloud of the object.
6.  **Predict Grasps**: Calls `client.predict_grasps` with the cropped point cloud data. The returned grasps are in the camera's coordinate frame.
7.  **Transform Grasps**: Uses `env.transform_grasps_to_robot_frame` (which internally uses `transform.py`) to convert the predicted grasps from the camera frame to the robot's base frame.
8.  **Filter Grasps**: Calls `client.filter_grasps` with the robot-frame grasps to identify reachable ones.
9.  **Visualize Valid Grasps**: Uses `vis.visualize_grasps` to display the *valid* grasps (transformed back to camera frame for visualization consistency) overlaid on the *cropped* point cloud.
10. **Select Grasp**: Chooses the first valid grasp returned by the filtering service.
11. **Execute Grasp**: Converts the chosen grasp pose (rotation matrix to Euler angles) and calls `env.execute_grasp_sequence` to perform the pick-and-lift motion in the simulation.

## Simulation (`sim.py`)

The `sim.py` module provides the `Sim` base class and the `SimGrasp` derived class for the simulation environment.

**Key Features:**

*   Loads a robot from a URDF file.
*   Loads objects (duck, tray, cube by default) into the scene.
*   Configures and renders data (color, depth, segmentation) from a simulated camera (defined in `cameras.py`).
*   Generates Open3D point clouds from camera data.
*   Provides robot control via Inverse Kinematics (`robot_control`).
*   Includes a pre-defined grasp execution sequence (`execute_grasp_sequence`).
*   Handles coordinate transformations between camera and robot frames (`transform_grasps_to_robot_frame`).

**Customization:**

*   **Objects**:
    *   Modify the `add_object` calls within the `SimGrasp.__init__` method in `sim.py`. You can change the URDF file paths, initial positions, orientations, and scaling factors for each object.
    *   Add or remove objects by adding/removing `add_object` calls.
*   **Camera**:
    *   Modify camera properties (pose, intrinsics, resolution, noise) in `cameras.py`.
    *   Select which camera configuration to use by changing which element of `cameras.RealSenseD435.CONFIG` (or another camera class) is accessed in `SimGrasp.__init__` and `render_camera`.
*   **Grasp Execution**:
    *   Modify the sequence of movements (e.g., approach strategy, lift height) within the `execute_grasp_sequence` method in `sim.py`.
    *   Change the robot control parameters (e.g., IK solver settings, joint control forces) in the `robot_control` method.

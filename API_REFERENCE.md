## Client API (`client.py`)

The `GeneralBionixClient` class provides methods to interact with the external services:

*   **`__init__(self, api_key: str)`**: Initializes the client with the URL for the services, HTTP headers, and user's API key.
*   **`crop_point_cloud(self, pcd: o3d.geometry.PointCloud, x: int, y: int) -> PointCloudData`**:
    *   Takes an Open3D point cloud and user-selected pixel coordinates (x, y).
    *   NOTE: The pointcloud data is expected in flattened RGB-D order. See `create_pointcloud()` in `sim.py` for an example.
    *   Sends the point cloud data and coordinates to the Point Cloud Cropping service.
    *   Returns the cropped point cloud of the object at the x, y coordinate (`PointCloudData` Pydantic model).
*   **`predict_grasps(self, cropped_pcd_data: PointCloudData) -> GraspsPredictionResponse`**:
    *   Takes the (potentially cropped) point cloud data.
    *   Sends it to the Grasp Prediction service.
    *   Returns a list of predicted grasps (`Grasp` Pydantic model) in the camera's coordinate frame, wrapped in a `GraspsPredictionResponse`.
*   **`filter_grasps(self, grasps: List[Grasp]) -> GraspsFilteringResponse`**:
    *   Takes a list of grasps (expected to be in the robot's base frame).
    *   Sends them to the Grasp Filtering service to check for reachability.
    *   Returns a `GraspsFilteringResponse` containing the indices of the valid grasps within the input list and optionally, the corresponding joint angles.

The script also defines Pydantic models (`Grasp`, `PointCloudData`, `PointCloudCropRequest`, etc.) used for structuring request and response data exchanged with the services.

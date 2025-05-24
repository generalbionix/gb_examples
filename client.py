from typing import List, Optional
from pydantic import BaseModel
import requests
import numpy as np
import open3d as o3d
import json

HEADERS = {"Content-Type": "application/json"}



class Grasp(BaseModel):
    rotation: List[List[float]]
    translation: List[float]


class PointCloudData(BaseModel):
    """
    NOTE: The pointcloud data is expected in flattened RGB-D order.
    """
    points: List[List[float]]
    colors: Optional[List[List[float]]] = None


class PointCloudCropRequest(BaseModel):
    pcd_data: PointCloudData
    x: int
    y: int


class GraspsPredictionResponse(BaseModel):
    grasps: List[Grasp]


class GraspPredictionRequest(BaseModel):
    pcd_data: PointCloudData


class GraspsFilteringRequest(BaseModel):
    grasps: List[Grasp]


class GraspsFilteringResponse(BaseModel):
    valid_grasp_idxs: List[int]
    valid_grasp_joint_angles: List[List[float]]


def call_gateway_service(service_name, payload, api_key):
    """
    Makes a call to the gateway API.
    
    Args:
        service_name (str): Name of the service to call (must exist in services.json)
        payload (dict): The data to send to the service
        api_key (str): Valid API key to be used as Bearer token
    
    Returns:
        Response object from the service
    """
    url = f"{service_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response


class GeneralBionixClient:
    def __init__(self, api_key: str):
        """
        Initializes the GeneralBionixClient.

        Sets up the URLs for the point cloud processing, grasp prediction,
        and grasp filtering services, along with default HTTP headers.
        """
        self.headers = HEADERS
        self.api_key = api_key
        self.headers["Authorization"] = f"Bearer {self.api_key}"


    def crop_point_cloud(self, pcd: o3d.geometry.PointCloud, x: int, y: int) -> PointCloudData:
        """
        Sends a request to the point cloud processing service to crop the input point cloud based on the given x and y coordinates.

        Args:
            pcd (o3d.geometry.PointCloud): The original point cloud to be cropped.
            x (int): The x-coordinate for the cropping operation (specific meaning depends on the service implementation).
            y (int): The y-coordinate for the cropping operation (specific meaning depends on the service implementation).

        Returns:
            PointCloudData: The data representing the cropped point cloud, containing points and optional colors.
        """
        pcd_data = PointCloudData(
            points=np.asarray(pcd.points).tolist(),
            colors=np.asarray(pcd.colors).tolist(),
        )
        pcd_request = PointCloudCropRequest(pcd_data=pcd_data, x=x, y=y).model_dump(exclude_defaults=True)
        response = call_gateway_service("https://gb-services--pcd-grasp-pipeline-fastapi-app-entry.modal.run/process_item/", pcd_request, self.api_key)
        response.raise_for_status()
        response_data = response.json()
        return PointCloudData(**response_data)


    def predict_grasps(self, cropped_pcd_data: PointCloudData) -> GraspsPredictionResponse:
        """
        Sends a request to the grasp prediction service to generate potential grasps for a cropped point cloud.

        Args:
            cropped_pcd_data (PointCloudData): The point cloud data (potentially cropped) for which to predict grasps.

        Returns:
            GraspsPredictionResponse: A response object containing a list of predicted grasps.
        """
        request = GraspPredictionRequest(pcd_data=cropped_pcd_data).model_dump(exclude_defaults=True)
        response = call_gateway_service("https://gb-services--grasp-service-fastapi-app-entry.modal.run/get_grasps/", request, self.api_key)
        response.raise_for_status()
        grasps_response_data = response.json()
        return GraspsPredictionResponse(**grasps_response_data)


    def filter_grasps(self, grasps: List[Grasp]) -> GraspsFilteringResponse:
        """Sends a request to filter grasps based on reachability."""
        """
        Sends a request to the grasp filtering service filter grasps based on reachability.

        Args:
            grasps (List[Grasp]): A list of grasp objects to be filtered.

        Returns:
            GraspsFilteringResponse: A response object containing the indices of the valid grasps
                                     and optionally corresponding valid joint angles.
        """
        request_obj = GraspsFilteringRequest(grasps=grasps).model_dump(exclude_defaults=True)
        response = call_gateway_service("https://gb-services--grasp-filtering-service-fastapi-app-entry.modal.run/process_item/", request_obj, self.api_key)
        response.raise_for_status()
        print("Request successful. Parsing response...")
        response_data = response.json()
        return GraspsFilteringResponse(**response_data)
"""
Cached wrapper for GeneralBionixClient using joblib.Memory for robust caching.

This module provides a caching layer using joblib.Memory, which is specifically
designed for caching expensive function calls in scientific computing contexts.
It handles complex data structures like numpy arrays and point clouds very well.
"""

from typing import List
import numpy as np
import open3d as o3d
from joblib import Memory

from client import GeneralBionixClient, PointCloudData, Grasp, GraspsPredictionResponse, GraspFilteringResponse


class CachedGeneralBionixClient:
    """
    A caching wrapper around GeneralBionixClient using joblib.Memory.
    
    This client automatically caches expensive API calls using joblib's robust
    caching system, which is designed for scientific computing workflows.
    The cache is persistent across program runs and uses content-based hashing
    to ensure identical inputs always return cached results.
    
    Attributes:
        client (GeneralBionixClient): The underlying API client
        enable_cache (bool): Whether caching is enabled
        memory (joblib.Memory): The joblib memory object for caching
    """
    
    def __init__(self, api_key: str, cache_dir: str = "./cache", enable_cache: bool = True, verbose: int = 1):
        """
        Initialize the cached client.
        
        Args:
            api_key (str): API key for the GeneralBionix services
            cache_dir (str, optional): Directory to store cache files. 
                Will be created if it doesn't exist. Defaults to "./cache".
            enable_cache (bool, optional): Whether to use caching. Set to False
                for debugging or to force fresh API calls. Defaults to True.
            verbose (int, optional): Verbosity level for joblib.Memory output.
                0 = silent, 1 = normal, 2 = verbose. Defaults to 1.
        """
        self.client = GeneralBionixClient(api_key)
        self.enable_cache = enable_cache
        
        if self.enable_cache:
            # Initialize joblib Memory with the cache directory
            self.memory = Memory(cache_dir, verbose=verbose)
            print(f"📁 Cache enabled: storing results in {cache_dir}")
            
            # Create cached versions of the API methods
            self._cached_crop_point_cloud = self.memory.cache(self._crop_point_cloud)
            self._cached_predict_grasps = self.memory.cache(self._predict_grasps)
            self._cached_filter_grasps = self.memory.cache(self._filter_grasps)
        else:
            print("⚠️  Cache disabled: all API calls will be made fresh")

    def _crop_point_cloud(self, points_array: np.ndarray, colors_array: np.ndarray, x: int, y: int) -> dict:
        """
        Internal method to crop point cloud - cached by joblib.
        
        This method handles the actual API call for point cloud cropping.
        It converts numpy arrays back to Open3D format for the API call
        and returns a dictionary for JSON serialization compatibility.
        
        Args:
            points_array (np.ndarray): Point cloud points as numpy array
            colors_array (np.ndarray): Point cloud colors as numpy array  
            x (int): X coordinate for cropping
            y (int): Y coordinate for cropping
            
        Returns:
            dict: Cropped point cloud data as dictionary for caching
        """
        # Convert arrays back to Open3D point cloud for the API call
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_array)
        pcd.colors = o3d.utility.Vector3dVector(colors_array)
        
        print("📡 Calling Point Cloud Cropping service...")
        result = self.client.crop_point_cloud(pcd, x, y)
        
        # Convert to dict for JSON serialization
        return result.model_dump()

    def _predict_grasps(self, pcd_points: List[List[float]], pcd_colors: List[List[float]]) -> dict:
        """
        Internal method to predict grasps - cached by joblib.
        
        This method handles the actual API call for grasp prediction.
        It reconstructs the PointCloudData object from the input lists
        and returns a dictionary for JSON serialization compatibility.
        
        Args:
            pcd_points (List[List[float]]): Point cloud points as nested list
            pcd_colors (List[List[float]]): Point cloud colors as nested list
            
        Returns:
            dict: Grasp prediction response as dictionary for caching
        """
        cropped_pcd_data = PointCloudData(points=pcd_points, colors=pcd_colors)
        
        print("📡 Calling Grasp Prediction service...")
        result = self.client.predict_grasps(cropped_pcd_data)
        
        # Convert to dict for JSON serialization
        return result.model_dump()

    def _filter_grasps(self, grasps_data: List[dict], robot_name: str) -> dict:
        """
        Internal method to filter grasps - cached by joblib.
        
        This method handles the actual API call for grasp filtering.
        It reconstructs Grasp objects from the input dictionaries
        and returns a dictionary for JSON serialization compatibility.
        
        Args:
            grasps_data (List[dict]): List of grasp dictionaries
            
        Returns:
            dict: Grasp filtering response as dictionary for caching
        """
        # Convert dict back to Grasp objects for the API call
        grasps = [Grasp(**grasp_dict) for grasp_dict in grasps_data]
        
        print("📡 Calling Grasp Filtering service...")
        result = self.client.filter_grasps(grasps, robot_name=robot_name)
        
        # Convert to dict for JSON serialization
        return result.model_dump()

    def crop_point_cloud(self, pcd: o3d.geometry.PointCloud, x: int, y: int) -> PointCloudData:
        """
        Crop point cloud with caching support.
        
        Identical point clouds and click coordinates will return cached results
        instantly without making additional API calls. The cache key is based
        on the content of the point cloud data and the coordinates.
        
        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud to crop
            x (int): X coordinate for cropping center
            y (int): Y coordinate for cropping center
            
        Returns:
            PointCloudData: Cropped point cloud data
        """
        if not self.enable_cache:
            return self.client.crop_point_cloud(pcd, x, y)
        
        # Convert point cloud to numpy arrays for hashing
        points_array = np.asarray(pcd.points)
        colors_array = np.asarray(pcd.colors)
        
        # Call cached method
        result_dict = self._cached_crop_point_cloud(points_array, colors_array, x, y)
        
        # Convert back to PointCloudData
        return PointCloudData(**result_dict)

    def predict_grasps(self, cropped_pcd_data: PointCloudData) -> GraspsPredictionResponse:
        """
        Predict grasps with caching support.
        
        Identical point cloud data will return cached grasp predictions
        instantly without making additional API calls. The cache key is based
        on the content of the point cloud data.
        
        Args:
            cropped_pcd_data (PointCloudData): Cropped point cloud data for grasp prediction
            
        Returns:
            GraspsPredictionResponse: Response containing predicted grasps and metadata
        """
        if not self.enable_cache:
            return self.client.predict_grasps(cropped_pcd_data)
        
        # Call cached method
        result_dict = self._cached_predict_grasps(cropped_pcd_data.points, cropped_pcd_data.colors)
        
        # Convert back to GraspsPredictionResponse
        return GraspsPredictionResponse(**result_dict)

    def filter_grasps(self, grasps: List[Grasp], robot_name: str) -> GraspFilteringResponse:
        """
        Filter grasps with caching support.
        
        Identical grasp lists will return cached filtering results
        instantly without making additional API calls. The cache key is based
        on the content of all grasps in the input list.
        
        Args:
            grasps (List[Grasp]): List of grasps to filter
            
        Returns:
            GraspsFilteringResponse: Response containing filtered grasps and metadata
        """
        if not self.enable_cache:
            return self.client.filter_grasps(grasps, robot_name)
        
        # Convert grasps to dicts for hashing
        grasps_data = [grasp.model_dump() for grasp in grasps]
        
        # Call cached method
        result_dict = self._cached_filter_grasps(grasps_data, robot_name)
        
        # Convert back to GraspsFilteringResponse
        return GraspFilteringResponse(**result_dict)

    def clear_cache(self, method_name: str = None) -> None:
        """
        Clear cache files.
        
        This method allows you to clear the cache either completely or for
        specific methods. Useful when you want to force fresh API calls or
        when cache files become too large.
        
        Args:
            method_name (str, optional): If specified, only clear cache for this method.
                Valid options: 'crop', 'predict', 'filter'.
                If None, clears all cache files. Defaults to None.
        """
        if not self.enable_cache:
            print("Cache is disabled, nothing to clear")
            return
        
        if method_name is None:
            # Clear all cache
            self.memory.clear()
            print("🗑️  Cleared all cache files")
        else:
            # Clear specific method cache
            method_map = {
                'crop': self._cached_crop_point_cloud,
                'predict': self._cached_predict_grasps, 
                'filter': self._cached_filter_grasps
            }
            
            if method_name in method_map:
                method_map[method_name].clear()
                print(f"🗑️  Cleared cache for {method_name} method")
            else:
                print(f"Unknown method: {method_name}. Options: 'crop', 'predict', 'filter'")

    def cache_stats(self) -> dict:
        """
        Get statistics about the cache.
        
        Provides information about cache usage including file count, total size,
        and cache directory location. Useful for monitoring cache growth and
        deciding when to clear cache files.
        
        Returns:
            dict: Dictionary containing cache statistics with the following keys:
                - cache_enabled (bool): Whether caching is enabled
                - cache_dir (str): Path to cache directory (if enabled)
                - total_files (int): Number of cache files (if enabled)
                - total_size_bytes (int): Total cache size in bytes (if enabled)
                - total_size_mb (float): Total cache size in MB (if enabled)
                - backend (str): Caching backend used
                - error (str): Error message if stats couldn't be retrieved
        """
        if not self.enable_cache:
            return {"cache_enabled": False}
        
        try:
            # Get cache directory info
            cache_dir = self.memory.location
            cache_path = self.memory.store_backend.location
            
            # Count files in cache directory
            import os
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(cache_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            return {
                "cache_enabled": True,
                "cache_dir": str(cache_path),
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "backend": "joblib.Memory"
            }
        except Exception as e:
            return {
                "cache_enabled": True,
                "error": f"Could not get cache stats: {e}",
                "backend": "joblib.Memory"
            } 
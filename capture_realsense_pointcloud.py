import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from open3d import open3d

WIDTH = 640
HEIGHT = 480



def capture_pointcloud() -> o3d.geometry.PointCloud:
    """
    Capture a pointcloud from the Realsense camera.

    Returns:
        pcd (open3d.geometry.PointCloud): The pointcloud
    """
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Turn ON Emitter
    if depth_sensor.supports(rs.option.emitter_always_on):
        depth_sensor.set_option(rs.option.emitter_always_on, 1)
    else:
        print("Note: emitter_always_on option not supported by this camera model")

    # Align
    align_to = rs.stream.color
    align = rs.align(align_to)

    pcd = open3d.geometry.PointCloud()
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    # Return pinhole camera intrinsics for Open3d
    intrinsics = aligned_frames.profile.as_video_stream_profile().intrinsics
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # Create point cloud directly from depth and color
    points = []
    colors = []
    # Get camera intrinsics
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    # Create points for every pixel
    for v in range(HEIGHT):
        for u in range(WIDTH):
            # Get depth value
            z = depth_image[v, u] * depth_scale
            # Convert pixel coordinates to 3D point
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(color_image[v, u] / 255.0)  # Normalize colors to [0,1]
    # Create Open3D point cloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    # Verify we have the correct number of points
    assert len(np.asarray(pcd.points)) == WIDTH * HEIGHT, (
        f"Expected {WIDTH * HEIGHT} points, got {len(np.asarray(pcd.points))}"
    )
    # TODO: we fixed this before so a flip is not needed. Why is this needed now?
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


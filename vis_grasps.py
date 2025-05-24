"""
This file is for visualizing the predicted grasps in the  meshcat 3D viewer. 
This launches a local server that can be accessed at http://127.0.0.1:7000/static/ for viewing the grasps along with the point cloud. 

Usage:
    vis = launch_visualizer()
    vis_grasps_meshcat(vis, grasp_list, point_cloud)
"""
from typing import List, Optional, Union, Any
import copy
import numpy as np
import open3d as o3d
import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf

from client import Grasp
from transform import make_transform_mat, transform_grasps_inv

def is_rot_mat(M: np.ndarray, tol: float = 1e-4) -> bool:
    """
    Check whether a matrix is a valid rotation matrix.

    A rotation matrix must be orthonormal (R * R.T == I) and have unit
    determinant. A small numerical tolerance accounts for floating point
    inaccuracies.

    Args:
        M (np.ndarray): Candidate rotation matrix.
        tol (float, optional): Numerical tolerance for validation checks. Defaults to 1e-4.

    Returns:
        bool: True if the matrix satisfies the rotation properties, False otherwise.

    Raises:
        ValueError: If the input matrix is not square.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Rotation matrix must be square.")

    identity = np.eye(M.shape[0])
    orthonormal = np.linalg.norm(M @ M.T - identity) < tol
    unit_det = abs(np.linalg.det(M) - 1.0) < tol
    valid = orthonormal and unit_det

    if not valid:
        print("Matrix failed rotation matrix check:")
        print("R R.T =\n", M @ M.T)
        print("det =", np.linalg.det(M))

    return valid


def to_meshcat_tri_geometry(mesh: Any) -> g.TriangularMeshGeometry:
    """
    Converts a trimesh object to a MeshCat triangular mesh geometry.

    Args:
        mesh (Any): A trimesh.TriMesh object containing vertices and faces.

    Returns:
        g.TriangularMeshGeometry: MeshCat geometry object ready for visualization.
    """
    return meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces)


def render_mesh(
    vis: meshcat.Visualizer,
    name: str,
    mesh: Any,
    color: Optional[Union[List[int], np.ndarray]] = None,
    transform: Optional[np.ndarray] = None,
) -> None:
    """
    Visualizes a 3D mesh in the MeshCat viewer.

    Args:
        vis (meshcat.Visualizer): The MeshCat visualizer instance.
        name (str): Unique identifier for this mesh in the scene tree.
        mesh (Any): A trimesh.TriMesh object to visualize.
        color (Optional[Union[List[int], np.ndarray]], optional): RGB color values (0-255).
            If None, a random color is generated. Defaults to None.
        transform (Optional[np.ndarray], optional): 4x4 transformation matrix to apply
            to the mesh. Defaults to None.
    """
    if color is None:
        color = np.random.randint(low=0, high=256, size=3)

    mesh_vis = to_meshcat_tri_geometry(mesh)
    color_hex = rgb_to_hex(tuple(color))
    material = meshcat.geometry.MeshPhongMaterial(color=color_hex)
    vis[name].set_object(mesh_vis, material)

    if transform is not None:
        vis[name].set_transform(transform)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """
    Convert an RGB colour (0-255 each) to a hexadecimal string.

    Args:
        rgb (tuple[int, int, int]): RGB color tuple with values from 0-255.

    Returns:
        str: Hexadecimal color string in format '0xRRGGBB'.

    Example:
        >>> rgb_to_hex((255, 0, 0))
        '0xff0000'
    """
    return f"0x{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def launch_visualizer(clear: bool = True) -> meshcat.Visualizer:
    """
    Launch a MeshCat visualiser connected to a local server.

    Args:
        clear (bool, optional): If True, delete all existing geometries in the
            visualiser. This is handy when re-running scripts in the same
            session. Defaults to True.

    Returns:
        meshcat.Visualizer: Connected MeshCat visualizer instance.
    """
    print("Waiting for MeshCat server... did you forget to run `meshcat-server`?")
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    if clear:
        vis.delete()
    return vis


def draw_axes(
    vis: meshcat.Visualizer,
    name: str,
    h: float = 0.15,
    radius: float = 0.01,
    o: float = 1.0,
    T: Optional[np.ndarray] = None,
) -> None:
    """
    Add a red-green-blue coordinate frame triad to the MeshCat visualiser.

    Creates three colored cylinders representing the X (red), Y (green), and Z (blue) axes.

    Args:
        vis (meshcat.Visualizer): The MeshCat visualizer instance.
        name (str): Name prefix for this frame (should be unique).
        h (float, optional): Length of each axis cylinder. Defaults to 0.15.
        radius (float, optional): Cylinder radius. Defaults to 0.01.
        o (float, optional): Opacity value (0.0 to 1.0). Defaults to 1.0.
        T (Optional[np.ndarray], optional): Optional 4×4 transform matrix to apply
            to the whole triad. Defaults to None.

    Raises:
        ValueError: If T contains an invalid rotation matrix.
    """
    axes = [
        ("x", [0, 0, 1], 0xFF0000, 0),
        ("y", [0, 1, 0], 0x00FF00, 1),
        ("z", [1, 0, 0], 0x0000FF, 2),
    ]

    for axis_name, axis_vec, color, idx in axes:
        material = g.MeshLambertMaterial(color=color, reflectivity=0.8, opacity=o)
        vis[name][axis_name].set_object(
            g.Cylinder(height=h, radius=radius), material
        )

        rot = mtf.rotation_matrix(np.pi / 2.0, axis_vec)
        rot[idx, 3] = h / 2
        vis[name][axis_name].set_transform(rot)

    if T is not None:
        if not is_rot_mat(T[:3, :3]):
            raise ValueError("Attempted to visualise an invalid rotation matrix.")
        vis[name].set_transform(T)


def render_bbox(
    vis: meshcat.Visualizer,
    name: str,
    dims: Union[List[float], np.ndarray],
    T: Optional[np.ndarray] = None,
    color: List[int] = [255, 0, 0],
) -> None:
    """
    Visualize a 3D bounding box using a wireframe representation.

    Args:
        vis (meshcat.Visualizer): The MeshCat visualizer instance.
        name (str): Unique identifier for this bounding box in the scene tree.
        dims (Union[List[float], np.ndarray]): Shape (3,), dimensions of the
            bounding box [width, height, depth].
        T (Optional[np.ndarray], optional): 4x4 transformation matrix to apply
            to this geometry. Defaults to None.
        color (List[int], optional): RGB color values (0-255). Defaults to [255, 0, 0].
    """
    color_hex = rgb_to_hex(tuple(color))
    material = meshcat.geometry.MeshBasicMaterial(wireframe=True, color=color_hex)
    bbox = meshcat.geometry.Box(dims)
    vis[name].set_object(bbox, material)

    if T is not None:
        vis[name].set_transform(T)


def render_point_cloud(
    vis: meshcat.Visualizer,
    name: str,
    pc: Union[List, np.ndarray],
    color: Optional[Union[List[int], np.ndarray]] = None,
    transform: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> None:
    """
    Visualise a point cloud with optional per-point colours and transformation.

    Args:
        vis (meshcat.Visualizer): The MeshCat visualizer instance.
        name (str): Unique identifier for this point cloud in the scene tree.
        pc (Union[List, np.ndarray]): Point cloud data with shape (N, 3) or (H, W, 3).
        color (Optional[Union[List[int], np.ndarray]], optional): Color data with
            same shape as pc (0-255 scale) or single RGB tuple. If None, uses white.
            Defaults to None.
        transform (Optional[np.ndarray], optional): 4x4 homogeneous transformation
            matrix to apply. Defaults to None.
        **kwargs (Any): Additional arguments passed to the PointCloud constructor.
    """
    # Flatten H×W×3 images into N×3 arrays.
    pc = pc.reshape(-1, pc.shape[-1]) if pc.ndim == 3 else pc

    if color is None:
        col = np.ones_like(pc)
    else:
        col = np.asarray(color)
        if col.ndim == 1:
            # Expand single RGB value to match all points.
            col = np.repeat(col[None, :], pc.shape[0], axis=0)
        else:
            col = col.reshape(pc.shape)
        col = col.astype(np.float32) / 255.0

    vis[name].set_object(g.PointCloud(position=pc.T, color=col.T, **kwargs))

    if transform is not None:
        vis[name].set_transform(transform)


def render_robot(
    vis: meshcat.Visualizer,
    robot: Any,
    name: str = "robot",
    q: Optional[List[float]] = None,
    color: Optional[Union[List[int], np.ndarray]] = None,
) -> None:
    """
    Visualize a robot model in MeshCat with optional joint configuration and coloring.

    Args:
        vis (meshcat.Visualizer): The MeshCat visualizer instance.
        robot (Any): Robot object with link_map and physical_link_map attributes.
        name (str, optional): Base name for the robot in the scene tree. Defaults to "robot".
        q (Optional[List[float]], optional): Joint configuration to set before visualization.
            Defaults to None.
        color (Optional[Union[List[int], np.ndarray]], optional): Color specification for
            robot links. Can be single RGB values or per-link colors. Defaults to None.
    """
    if q is not None:
        robot.set_joint_cfg(q)
    robot_link_poses = {
        linkname: robot.link_poses[linkmesh][0].cpu().numpy()
        for linkname, linkmesh in robot.link_map.items()
    }
    if color is not None and isinstance(color, np.ndarray) and len(color.shape) == 2:
        assert color.shape[0] == len(robot.physical_link_map)
    link_id = -1
    for link_name in robot.physical_link_map:
        link_id += 1
        coll_mesh = robot.link_map[link_name].collision_mesh
        assert coll_mesh is not None
        link_color = None
        if color is not None and not isinstance(color, np.ndarray):
            color = np.asarray(color)
        if color.ndim == 1:
            link_color = color
        else:
            link_color = color[link_id]
        if coll_mesh is not None:
            render_mesh(
                vis[name],
                f"{link_name}_{robot}",
                coll_mesh,
                color=link_color,
                transform=robot_link_poses[link_name].astype(np.float),
            )


def get_grasp_points() -> np.ndarray:
    """
    Return a set of 3D points outlining an antipodal grasp pose for visualization.

    Generates control points representing the geometry of a typical parallel-jaw
    gripper configuration, including finger positions and central axis.

    Returns:
        np.ndarray: Shape (4, 7), homogeneous coordinates of grasp visualization points.
    """
    control = np.array(
        [
            [0.05268743, -0.00005996, 0.059, 1.0],
            [-0.05268743, 0.00005996, 0.059, 1.0],
            [0.05268743, -0.00005996, 0.10527314, 1.0],
            [-0.05268743, 0.00005996, 0.10527314, 1.0],
        ]
    )
    mid = (control[0] + control[1]) / 2

    grasp_pc = np.array(
        [control[-2], control[0], mid, [0, 0, 0, 1], mid, control[1], control[-1]],
        dtype=np.float32,
    ).T
    return grasp_pc


def render_grasp(
    vis: meshcat.Visualizer,
    name: str,
    transform: np.ndarray,
    color: List[int] = [255, 0, 0],
    **kwargs: Any,
) -> None:
    """
    Render a grasp pose as a simple line strip in MeshCat.

    Visualizes the grasp configuration using connected line segments that represent
    the gripper geometry and orientation.

    Args:
        vis (meshcat.Visualizer): The MeshCat visualizer instance.
        name (str): Unique identifier for this grasp visualization in the scene tree.
        transform (np.ndarray): 4x4 transformation matrix defining the grasp pose.
        color (List[int], optional): RGB color values (0-255). Defaults to [255, 0, 0].
        **kwargs (Any): Additional arguments passed to the MeshBasicMaterial constructor.
    """
    grasp_vertices = get_grasp_points()
    vis[name].set_object(
        g.Line(
            g.PointsGeometry(grasp_vertices),
            g.MeshBasicMaterial(color=rgb_to_hex(tuple(color)), **kwargs),
        )
    )
    vis[name].set_transform(transform.astype(np.float64))


def vis_grasps_meshcat(
    vis: meshcat.Visualizer, 
    grasps: List[Grasp], 
    pcd: o3d.geometry.PointCloud
) -> None:
    """
    Visualize grasp poses and point cloud data in MeshCat.

    Displays a 3D scene containing the input point cloud and all provided grasp
    poses. Also shows a camera coordinate frame for reference.

    Args:
        vis (meshcat.Visualizer): The MeshCat visualizer instance.
        grasps (List[Grasp]): List of grasp poses to visualize.
        pcd (o3d.geometry.PointCloud): Point cloud data containing the scene geometry.
    """
    grasps_inv = transform_grasps_inv(copy.deepcopy(grasps))
    rgb = (np.asarray(pcd.colors) * 255).astype('uint8')
    xyz = np.asarray(pcd.points)
    cam_pose = make_transform_mat()
    draw_axes(vis, 'camera', T=cam_pose)
    for i in range(len(grasps_inv)):
        g = np.eye(4)
        g[:3, :3] = np.array(grasps_inv[i].rotation)
        g[:3, 3] = np.array(grasps_inv[i].translation)
        render_point_cloud(vis, 'scene', xyz, rgb, size=0.005)
        render_grasp(
            vis, f"object_0/grasps/{i}",
            g, linewidth=0.2
        )
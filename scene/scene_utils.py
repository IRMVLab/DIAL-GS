from typing import NamedTuple
import numpy as np
from utils.graphics_utils import getWorld2View2
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    est_depth_map: np.array = None
    gt_depth_map: np.array = None
    sky_mask: np.array = None
    semantic_mask: np.array = None
    timestamp: float = 0.0
    FovY: float = None
    FovX: float = None
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None
    pointcloud_camera: np.array = None
    instance_mask: np.array = None
    sam_mask: np.array = None
    dynamic_mask: np.array = None
    id_mask: np.array = None  
    feat_map: np.array = None
    normal_map: np.array = None
    objects: np.array = None
    intrinsic: np.array = None
    c2w: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    time_interval: float = 0.02
    time_duration: list = [0, 1]
    dynamic_dict: dict = None
    full_cameras : list = None
    bg_point_cloud: BasicPointCloud = None
    bg_ply_path: str = None
    cam_frustum_aabb: np.array = None
    num_panoptic_objects: int = 0
    panoptic_id_to_idx: dict = None
    panoptic_object_ids: list = None
    occ_grid: np.array = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    if 'time' in vertices:
        timestamp = vertices['time'][:, None]
    else:
        timestamp = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals, time=timestamp)


def storePly(path, xyz, rgb, timestamp=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('time', 'f4')]

    normals = np.zeros_like(xyz)
    if timestamp is None:
        timestamp = np.zeros_like(xyz[:, :1])

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, timestamp), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

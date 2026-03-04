#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
import random
from matplotlib import cm
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay
import faiss
import time

def GridSample3D(in_pc,in_shs, voxel_size=0.013):
    in_pc_ = in_pc[:,:3].copy()
    quantized_pc = np.around(in_pc_ / voxel_size)
    quantized_pc -= np.min(quantized_pc, axis=0)
    pc_boundary = np.max(quantized_pc, axis=0) - np.min(quantized_pc, axis=0)
    
    voxel_index = quantized_pc[:,0] * pc_boundary[1] * pc_boundary[2] + quantized_pc[:,1] * pc_boundary[2] + quantized_pc[:,2]
    
    split_point, index = get_split_point(voxel_index)
    
    in_points = in_pc[index,:]
    out_points = in_points[split_point[:-1],:]
    
    in_colors = in_shs[index]
    out_colors = in_colors[split_point[:-1]]
    
    return out_points,out_colors

def get_split_point(labels):
    index = np.argsort(labels)
    label = labels[index]
    label_shift = label.copy()
    
    label_shift[1:] = label[:-1]
    remain = label - label_shift
    step_index = np.where(remain > 0)[0].tolist()
    step_index.insert(0,0)
    step_index.append(labels.shape[0])
    return step_index,index

def sample_on_aabb_surface(aabb_center, aabb_size, n_pts=1000, above_half=False):
    """
    0:立方体的左面(x轴负方向)
    1:立方体的右面(x轴正方向)
    2:立方体的下面(y轴负方向)
    3:立方体的上面(y轴正方向)
    4:立方体的后面(z轴负方向)
    5:立方体的前面(z轴正方向)
    """
    # Choose a face randomly
    faces = np.random.randint(0, 6, size=n_pts)

    # Generate two random numbers
    r_ = np.random.random((n_pts, 2))

    # Create an array to store the points
    points = np.zeros((n_pts, 3))

    # Define the offsets for each face
    offsets = np.array([
        [-aabb_size[0]/2, 0, 0],
        [aabb_size[0]/2, 0, 0],
        [0, -aabb_size[1]/2, 0],
        [0, aabb_size[1]/2, 0],
        [0, 0, -aabb_size[2]/2],
        [0, 0, aabb_size[2]/2]
    ])

    # Define the scales for each face
    scales = np.array([
        [aabb_size[1], aabb_size[2]],
        [aabb_size[1], aabb_size[2]],
        [aabb_size[0], aabb_size[2]],
        [aabb_size[0], aabb_size[2]],
        [aabb_size[0], aabb_size[1]],
        [aabb_size[0], aabb_size[1]]
    ])

    # Define the positions of the zero column for each face
    zero_column_positions = [0, 0, 1, 1, 2, 2]
    # Define the indices of the aabb_size components for each face
    aabb_size_indices = [[1, 2], [1, 2], [0, 2], [0, 2], [0, 1], [0, 1]]
    # Calculate the coordinates of the points for each face
    for i in range(6):
        mask = faces == i
        r_scaled = r_[mask] * scales[i]
        r_scaled = np.insert(r_scaled, zero_column_positions[i], 0, axis=1)
        aabb_size_adjusted = np.insert(aabb_size[aabb_size_indices[i]] / 2, zero_column_positions[i], 0)
        points[mask] = aabb_center + offsets[i] + r_scaled - aabb_size_adjusted
        #visualize_points(points[mask], aabb_center, aabb_size)
    #visualize_points(points, aabb_center, aabb_size)
        
    # 提取上半部分的点
    if above_half:
        points = points[points[:, -1] > aabb_center[-1]]
    return points

def get_OccGrid(pts, aabb, occ_voxel_size):
    # 计算网格的大小
    grid_size = np.ceil((aabb[1] - aabb[0]) / occ_voxel_size).astype(int)
    assert pts.min() >= aabb[0].min() and pts.max() <= aabb[1].max(), "Points are outside the AABB"

    # 创建一个空的网格
    voxel_grid = np.zeros(grid_size, dtype=np.uint8)

    # 将点云转换为网格坐标
    grid_pts = ((pts - aabb[0]) / occ_voxel_size).astype(int)

    # 将网格中的点设置为1
    voxel_grid[grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2]] = 1

    # check
    #voxel_coords = np.floor((pts - aabb[0]) / occ_voxel_size).astype(int)
    #occ = voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]

    return voxel_grid

def visualize_depth(depth, near=0.2, far=13, linear=False):
    depth = depth[0].clone().detach().cpu().numpy()
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    if linear:
        curve_fn = lambda x: -x
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.) * 255
    out_depth = torch.from_numpy(out_depth).permute(2, 0, 1).float().cuda() / 255
    return out_depth

def feature_to_rgb(features):
    # Input features shape: (8, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T
    features_np = features_reshaped.detach().cpu().numpy()

    if features_np.size == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    total_var = np.var(features_np)
    if total_var < 1e-8:
        return np.zeros((H, W, 3), dtype=np.uint8)

    n_components = min(3, features_np.shape[1], features_np.shape[0])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_np)

    if n_components < 3:
        pad = np.zeros((pca_result.shape[0], 3 - n_components), dtype=pca_result.dtype)
        pca_result = np.concatenate([pca_result, pad], axis=1)

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_min = pca_result.min()
    denom = pca_result.max() - pca_min
    if denom < 1e-8:
        return np.zeros((H, W, 3), dtype=np.uint8)

    pca_normalized = 255 * (pca_result - pca_min) / denom

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id_to_rgb(ids):
    """
    接受一张(H,W)的张量，每个数字表示一个ID，把ID映射为一个固定颜色后，返回一张(3,H,W)的RGB图像
    超快速版本 - 完全向量化
    """
    device = ids.device
    H, W = ids.shape
    
    # 扁平化处理
    ids_flat = ids.view(-1)  # (H*W,)
    
    # 使用ID直接计算颜色 - 完全向量化
    hue = (ids_flat.float() * 137.508) % 360  # (H*W,)
    
    # 使用位运算生成确定性的饱和度和亮度
    hash_vals = (ids_flat * 42) % 1000000
    saturation = 0.7 + 0.3 * ((hash_vals % 1000).float() / 1000.0)
    value = 0.8 + 0.2 * (((hash_vals // 1000) % 1000).float() / 1000.0)
    
    # 向量化HSV转RGB
    c = value * saturation
    hue_segment = hue / 60.0
    x = c * (1 - torch.abs(hue_segment % 2 - 1))
    m = value - c
    
    # 创建RGB张量
    rgb_flat = torch.zeros((ids_flat.shape[0], 3), device=device)
    
    # 使用掩码向量化处理每个色调段
    segments = (hue_segment.long() % 6)
    
    mask0 = (segments == 0)
    mask1 = (segments == 1)  
    mask2 = (segments == 2)
    mask3 = (segments == 3)
    mask4 = (segments == 4)
    mask5 = (segments == 5)
    
    rgb_flat[mask0] = torch.stack([c[mask0], x[mask0], torch.zeros_like(c[mask0])], dim=1)
    rgb_flat[mask1] = torch.stack([x[mask1], c[mask1], torch.zeros_like(c[mask1])], dim=1)
    rgb_flat[mask2] = torch.stack([torch.zeros_like(c[mask2]), c[mask2], x[mask2]], dim=1)
    rgb_flat[mask3] = torch.stack([torch.zeros_like(c[mask3]), x[mask3], c[mask3]], dim=1)
    rgb_flat[mask4] = torch.stack([x[mask4], torch.zeros_like(c[mask4]), c[mask4]], dim=1)
    rgb_flat[mask5] = torch.stack([c[mask5], torch.zeros_like(c[mask5]), x[mask5]], dim=1)
    
    # 添加偏移
    rgb_flat = rgb_flat + m.view(-1, 1)
    
    # 限制范围
    rgb_flat = torch.clamp(rgb_flat, 0, 1)
    
    # 重塑为图像形状
    rgb_image = rgb_flat.view(H, W, 3)
    
    return rgb_image.permute(2,0,1).float()

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_step_lr_func(lr_init, lr_final, start_step):
    def helper(step):
        if step < start_step:
            return lr_init
        else:
            return lr_final
    return helper

def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def rotation_to_quaternion(R):
    r11, r12, r13 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r21, r22, r23 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r31, r32, r33 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    qw = torch.sqrt((1 + r11 + r22 + r33).clamp_min(1e-7)) / 2
    qx = (r32 - r23) / (4 * qw)
    qy = (r13 - r31) / (4 * qw)
    qz = (r21 - r12) / (4 * qw)

    quaternion = torch.stack((qw, qx, qy, qz), dim=-1)
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    return quaternion

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    r11 = 1 - 2 * y * y - 2 * z * z
    r12 = 2 * x * y - 2 * w * z
    r13 = 2 * x * z + 2 * w * y

    r21 = 2 * x * y + 2 * w * z
    r22 = 1 - 2 * x * x - 2 * z * z
    r23 = 2 * y * z - 2 * w * x

    r31 = 2 * x * z - 2 * w * y
    r32 = 2 * y * z + 2 * w * x
    r33 = 1 - 2 * x * x - 2 * y * y

    rotation_matrix = torch.stack((torch.stack((r11, r12, r13), dim=1),
                                   torch.stack((r21, r22, r23), dim=1),
                                   torch.stack((r31, r32, r33), dim=1)), dim=1)
    return rotation_matrix

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result_quaternion = torch.stack((w, x, y, z), dim=1)
    return result_quaternion

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import logging

import sys

def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][%(levelname)s] - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
 
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)



def compute_camera_frustum_corners(depth_map: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """ Computes the 3D coordinates of the camera frustum corners based on the depth map, pose, and intrinsics.
    Args:
        depth_map: The depth map of the scene.
        pose: The camera pose matrix.
        intrinsics: The camera intrinsic matrix.
    Returns:
        An array of 3D coordinates for the frustum corners.
    """
    height, width = depth_map.shape
    depth_map = depth_map[depth_map > 0]
    min_depth, max_depth = depth_map.min(), depth_map.max()
    corners = np.array(
        [
            [0, 0, min_depth],
            [width, 0, min_depth],
            [0, height, min_depth],
            [width, height, min_depth],
            [0, 0, max_depth],
            [width, 0, max_depth],
            [0, height, max_depth],
            [width, height, max_depth],
        ]
    )
    x = (corners[:, 0] - intrinsics[0, 2]) * corners[:, 2] / intrinsics[0, 0]
    y = (corners[:, 1] - intrinsics[1, 2]) * corners[:, 2] / intrinsics[1, 1]
    z = corners[:, 2]
    corners_3d = np.vstack((x, y, z, np.ones(x.shape[0]))).T
    corners_3d = pose @ corners_3d.T
    return corners_3d.T[:, :3] 

def compute_frustum_aabb(frustum_corners: torch.Tensor):
    """ Computes a mask indicating which points lie inside a given axis-aligned bounding box (AABB).
    Args:
        points: An array of 3D points.
        min_corner: The minimum corner of the AABB.
        max_corner: The maximum corner of the AABB.
    Returns:
        A boolean array indicating whether each point lies inside the AABB.
    """
    return torch.min(frustum_corners, axis=0).values, torch.max(frustum_corners, axis=0).values

def points_inside_aabb_mask(points: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    """ Computes a mask indicating which points lie inside the camera frustum.
    Args:
        points: A tensor of 3D points.
        frustum_planes: A tensor representing the planes of the frustum.
    Returns:
        A boolean tensor indicating whether each point lies inside the frustum.
    """
    return (
        (points[:, 0] >= min_corner[0])
        & (points[:, 0] <= max_corner[0])
        & (points[:, 1] >= min_corner[1])
        & (points[:, 1] <= max_corner[1])
        & (points[:, 2] >= min_corner[2])
        & (points[:, 2] <= max_corner[2]))

def points_inside_frustum_mask(points: torch.Tensor, frustum_planes: torch.Tensor) -> torch.Tensor:
    """ Computes a mask indicating which points lie inside the camera frustum.
    Args:
        points: A tensor of 3D points.
        frustum_planes: A tensor representing the planes of the frustum.
    Returns:
        A boolean tensor indicating whether each point lies inside the frustum.
    """
    num_pts = points.shape[0]
    ones = torch.ones(num_pts, 1).to(points.device)
    plane_product = torch.cat([points, ones], axis=1) @ frustum_planes.T
    return torch.all(plane_product <= 0, axis=1)

def compute_camera_frustum_planes(frustum_corners: np.ndarray) -> torch.Tensor:
    """ Computes the planes of the camera frustum from its corners.
    Args:
        frustum_corners: An array of 3D coordinates representing the corners of the frustum.

    Returns:
        A tensor of frustum planes.
    """
    # near, far, left, right, top, bottom
    planes = torch.stack(
        [
            torch.cross(
                frustum_corners[2] - frustum_corners[0],
                frustum_corners[1] - frustum_corners[0]
            ),
            torch.cross(
                frustum_corners[5] - frustum_corners[4],
                frustum_corners[6] - frustum_corners[4]
                
            ),
            torch.cross(
                frustum_corners[4] - frustum_corners[0],
                frustum_corners[2] - frustum_corners[0]
            ),
            torch.cross(
                frustum_corners[7] - frustum_corners[3],
                frustum_corners[1] - frustum_corners[3]
            ),
            torch.cross(
                frustum_corners[5] - frustum_corners[1], 
                frustum_corners[0] - frustum_corners[1]
            ),
            torch.cross(
                frustum_corners[6] - frustum_corners[2], 
                frustum_corners[3] - frustum_corners[2]
            ),
        ]
    )

    corresponding_points = torch.stack(
        [
            frustum_corners[0],  # near left top
            frustum_corners[4],  # far left top
            frustum_corners[2],  # near left bottom
            frustum_corners[7],  # far right bottom
            frustum_corners[1],  # near right top
            frustum_corners[6],  # far left bottom
        ]
    )

    # planes是一个 (6, 3) 的张量，每一行是一个平面的法向量
    D = torch.stack([-torch.dot(plane, corresponding_points[i]) for i, plane in enumerate(planes)])
    return torch.cat([planes, D[:, None]], dim=1).float()


def compute_frustum_point_ids(pts: torch.Tensor, frustum_corners: torch.Tensor, device: str = "cuda"):
    """ Identifies points within the camera frustum, optimizing for computation on a specified device.
    Args:
        pts: A tensor of 3D points.
        frustum_corners: A tensor of 3D coordinates representing the corners of the frustum.
        device: The computation device ("cuda" or "cpu").
    Returns:
        Indices of points lying inside the frustum.
    """
    if pts.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64, device=device)
    # Broad phase
    pts = pts.to(device)
    frustum_corners = frustum_corners.to(device)

    min_corner, max_corner = compute_frustum_aabb(frustum_corners)
    inside_aabb_mask = points_inside_aabb_mask(pts, min_corner, max_corner) # (N,) bool
    # print(f"Existed points inside AABB: {inside_aabb_mask.sum().item()}")

    # Narrow phase
    frustum_planes = compute_camera_frustum_planes(frustum_corners)
    frustum_planes = frustum_planes.to(device)
    inside_frustum_mask = points_inside_frustum_mask(pts[inside_aabb_mask], frustum_planes) # (M,)
    # print(f"Existed points inside frustum: {inside_frustum_mask.sum().item()}")

    inside_aabb_mask[inside_aabb_mask == 1] = inside_frustum_mask # （
    return inside_aabb_mask #(N,) bool


def points_inside_convex_hull(point_cloud, mask, remove_outliers=False, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask


def points_inside_knn_region(point_cloud, mask_A, mask_B, k=5):
    """
    为mask_A中的每个点在mask_B中寻找k个最近邻居，并在结果mask中标记这些邻居为True
    使用FAISS加速KNN搜索 - 完全向量化的高效实现
    
    参数:
        point_cloud (torch.Tensor): 形状为(N, 3)的点云坐标
        mask_A (torch.Tensor): 形状为(N,)的布尔tensor，指示源点集
        mask_B (torch.Tensor): 形状为(N,)的布尔tensor，指示目标点集（在其中寻找近邻）
        k (int): 每个源点需要查找的近邻数量
    
    返回:
        torch.Tensor: 形状为(N,)的布尔tensor，标记了所有找到的近邻点
    """

    start_time = time.time()
    device = point_cloud.device
    
    # 提取A和B中的点
    points_A = point_cloud[mask_A].cpu().numpy().astype('float32')  # 源点集 (~5万点)
    points_B = point_cloud[mask_B].cpu().numpy().astype('float32')  # 目标点集 (~100万点)
    
    print(f"提取点耗时: {time.time() - start_time:.4f}秒")
    index_start = time.time()
    
    # 如果A或B为空，则返回全False的mask
    if len(points_A) == 0 or len(points_B) == 0:
        return torch.zeros(len(point_cloud), dtype=torch.bool, device=device)
    
    # 构建FAISS索引
    d = points_B.shape[1]  # 维度 (3)
    k = min(k, len(points_B))  # 确保k不大于B中的点数
    
    # 对于100万级别的B点集，使用更高效的HNSW索引
    use_gpu = torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
    
    if use_gpu:

        # 使用GPU版本
        print("Using FAISS with GPU")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
        index.add(points_B)
    else:
        # 对于CPU，尝试更高效的索引
        if len(points_B) > 100000:
            # 大数据集使用IVF
            nlist = 1024  # 聚类中心数量
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            index.train(points_B)
            index.add(points_B)
            index.nprobe = 32  # 搜索参数
        else:
            # 小数据集直接使用精确搜索
            index = faiss.IndexFlatL2(d)
            index.add(points_B)
    
    print(f"构建索引耗时: {time.time() - index_start:.4f}秒")
    search_start = time.time()
    
    # 批量查询以减少开销
    batch_size = 10000  # 增大批次以提高效率
    indices_list = []
    
    for i in range(0, len(points_A), batch_size):
        batch_points = points_A[i:i+batch_size]
        _, batch_indices = index.search(batch_points, k)
        indices_list.append(batch_indices)
    
    # 合并结果
    indices = np.vstack(indices_list)
    
    print(f"搜索耗时: {time.time() - search_start:.4f}秒")
    post_start = time.time()
    
    # 向量化处理 - 核心优化点
    # 1. 首先获取所有有效的索引（过滤掉-1）
    mask = indices >= 0
    valid_indices = indices[mask]  # 一维数组
    
    # 2. 转换为全局索引 - 一次性操作而不是逐个处理
    global_B_indices = np.where(mask_B.cpu().numpy())[0]  # B中点在原始点云中的索引
    global_indices = global_B_indices[valid_indices]  # 一维数组
    
    # 3. 创建结果掩码并标记
    result_mask = torch.zeros(len(point_cloud), dtype=torch.bool, device=device)
    result_mask[torch.tensor(global_indices, device=device)] = True
    
    print(f"后处理耗时: {time.time() - post_start:.4f}秒")
    print(f"总耗时: {time.time() - start_time:.4f}秒")
    
    return result_mask

def remove_outliers_radius(points, radius=0.2, min_neighbors=5):
    """
    基于半径的邻域滤波
    
    Args:
        points: (N, 3) 动态点坐标
        radius: 搜索半径
        min_neighbors: 最小邻居数量
    """
    from sklearn.neighbors import NearestNeighbors
    
    points_np = points.cpu().numpy()
    nbrs = NearestNeighbors(radius=radius).fit(points_np)
    distances, indices = nbrs.radius_neighbors(points_np)
    
    # 计算每个点的邻居数量（排除自身）
    neighbor_counts = np.array([len(idx) - 1 for idx in indices])
    
    # 保留邻居数量足够的点
    inlier_mask = np.zeros(points_np.shape[0], dtype=bool)
    inlier_mask[neighbor_counts >= min_neighbors] = True
    
    return torch.tensor(inlier_mask, device=points.device)

def remove_outliers_center_distance(points, distance_factor=2.0):
    """
    基于到点云中心距离的离群点去除
    
    Args:
        points: (N, 3) 动态点坐标
        distance_factor: 距离倍数因子，越大保留的点越多
    
    Returns:
        inlier_mask: (N,) 布尔mask
    """
    if points.shape[0] < 4:
        return torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    
    # 计算点云中心（质心）
    center = points.mean(dim=0)  # (3,)
    
    # 计算每个点到中心的距离
    distances = torch.norm(points - center, dim=1)  # (N,)
    
    # 计算距离的统计量
    mean_distance = distances.mean()
    std_distance = distances.std()
    
    # 设定阈值：均值 + 倍数*标准差
    threshold = mean_distance + distance_factor * std_distance
    
    # 保留距离小于阈值的点
    inlier_mask = distances <= threshold
    
    return inlier_mask

def get_mask_from_projection(point_cloud, valid_mask, camera_intrinsics, camera_extrinsics, dynamic_mask, image_width, image_height):
    """
    将点云投影到像素平面上，找出落在dynamic mask为True区域的点（向量化实现）
    """
    device = point_cloud.device
    
    # 提取有效点
    valid_points = point_cloud[valid_mask]  # (V, 3)
    
    # 转换为齐次坐标
    homogeneous_points = torch.cat([valid_points, torch.ones(valid_points.shape[0], 1, device=device)], dim=1)  # (V, 4)
    
    # 世界坐标到相机坐标
    camera_points = torch.matmul(homogeneous_points, camera_extrinsics.t())  # (V, 4)
    
    # 只保留位于相机前方的点 (z > 0)
    front_mask = camera_points[:, 2] > 0
    camera_points = camera_points[front_mask]  # (F, 4)
    
    # 归一化为非齐次坐标
    normalized_points = camera_points[:, :3] / camera_points[:, 2:3]  # (F, 3)
    
    # 应用内参，投影到像素坐标
    pixel_points = torch.matmul(normalized_points, camera_intrinsics.t())  # (F, 3)
    pixel_coords = pixel_points[:, :2]  # (F, 2), [u, v]
    
    # 转换为整数坐标
    pixel_coords = pixel_coords.round().long()
    
    # 筛选出在图像范围内的点
    valid_pixel_mask = (
        (pixel_coords[:, 0] >= 0) &
        (pixel_coords[:, 0] < image_width) &
        (pixel_coords[:, 1] >= 0) &
        (pixel_coords[:, 1] < image_height)
    )
    
    # 生成结果掩码
    result_mask = torch.zeros(point_cloud.shape[0], dtype=torch.bool, device=device)
    
    # 计算索引链
    original_indices = torch.where(valid_mask)[0]
    front_indices = original_indices[front_mask]
    visible_indices = front_indices[valid_pixel_mask]
    
    # 获取有效的像素坐标
    valid_pixel_coords = pixel_coords[valid_pixel_mask]  # (P, 2)
    
    # 向量化方式检查像素是否在动态掩码区域
    # 注意：dynamic_mask的索引顺序是[v, u]
    v_coords = valid_pixel_coords[:, 1].clamp(0, dynamic_mask.shape[0]-1)
    u_coords = valid_pixel_coords[:, 0].clamp(0, dynamic_mask.shape[1]-1)
    
    # 批量获取mask值
    mask_values = dynamic_mask[v_coords, u_coords]
    
    # 标记所有满足条件的点
    result_mask[visible_indices[mask_values]] = True
    
    return result_mask

import sys
import os

# 动态添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
import numpy as np
import torch
from scene import GaussianModel, EnvLight
from scene.cameras import Camera
from utils.graphics_utils import BasicPointCloud
from gaussian_renderer import get_renderer
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import faiss

render_func, render_wrapper, render_func_merge, render_wrapper_merge= get_renderer('gs')


def sample_pixels_based_on_gradient(image: np.ndarray, num_samples: int) -> np.ndarray:
    """ Samples pixel indices based on the gradient magnitude of an image.
    Args:
        image: The image from which to sample pixels.
        num_samples: The number of pixels to sample.
    Returns:
        Indices of the sampled pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize the gradient magnitude to create a probability map
    prob_map = grad_magnitude / np.sum(grad_magnitude)

    # Flatten the probability map
    prob_map_flat = prob_map.flatten() # (H, W)

    # Sample pixel indices based on the probability map
    sampled_indices = np.random.choice(prob_map_flat.size, size=num_samples, p=prob_map_flat)  # 根据梯度采样像素
    return sampled_indices.T

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
    inside_frustum_mask = points_inside_frustum_mask(pts[inside_aabb_mask], frustum_planes) #
    # print(f"Existed points inside frustum: {inside_frustum_mask.sum().item()}")

    inside_aabb_mask[inside_aabb_mask == 1] = inside_frustum_mask
    return inside_aabb_mask #(N,) bool


def compute_new_points_ids(frustum_points: torch.Tensor, new_pts: torch.Tensor,
                           radius: float = 0.03, device: str = "cpu") -> torch.Tensor:
    """ Having newly initialized points, decides which of them should be added to the map.
        For every new point, if there are no neighbors within the radius in the frustum points,
        it is added to the submap.
    Args:
        frustum_points: Point within a current frustum of the active submap of shape (N, 3)
        new_pts: New 3D Gaussian means which are about to be added to the submap of shape (N, 3)
        radius: Radius whithin which the points are considered to be neighbors
        device: Execution device
    Returns:
        Indicies of the new points that should be added to the submap of shape (N)
    """
    if frustum_points.shape[0] == 0:
        return torch.arange(new_pts.shape[0])
    
    # 转换为numpy并检查数据范围
    if isinstance(frustum_points, torch.Tensor):
        frustum_points = frustum_points.detach().cpu().numpy()
    if isinstance(new_pts, torch.Tensor):
        new_pts = new_pts.detach().cpu().numpy()
    
   
    # 计算所有点的中心和缩放因子
    all_points = np.vstack([frustum_points, new_pts])
    center = np.mean(all_points, axis=0)
    scale = np.max(np.abs(all_points - center)) + 1e-6
    
    
    # 归一化坐标到[-1, 1]范围
    frustum_points_norm = (frustum_points - center) / scale
    new_pts_norm = (new_pts - center) / scale
    
    # 相应地缩放半径
    radius_norm = radius / scale
    
    # 使用归一化坐标进行FAISS搜索
    if device == "cpu":
        pts_index = faiss.IndexFlatL2(3)
    else:
        pts_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(3))
    
    frustum_points_norm = np.ascontiguousarray(frustum_points_norm.astype(np.float32))
    pts_index.add(frustum_points_norm)

    # 分批最近邻搜索
    split_size = 65535
    num_new_pts = new_pts_norm.shape[0]
    distances_list, ids_list = [], []
    
    for i in range(0, num_new_pts, split_size):
        end_idx = min(i + split_size, num_new_pts)
        batch_new_pts = new_pts_norm[i:end_idx]
        batch_new_pts = np.ascontiguousarray(batch_new_pts.astype(np.float32))
        
        # FAISS搜索
        distance, id = pts_index.search(batch_new_pts, 8) # 8表示返回8个最近邻
        distance_euclidean = np.sqrt(distance) * scale
        
        distances_list.append(torch.from_numpy(distance_euclidean).to(device))
        ids_list.append(torch.from_numpy(id).to(device))
    
    distances = torch.cat(distances_list, dim=0)  # (新点数, 8)
    ids = torch.cat(ids_list, dim=0)  # (新点数, 8)
    
    # 检查在半径内的邻居数
    within_radius = distances < radius  # (N, 8)
    neighbor_counts = within_radius.sum(axis=1).int()  # (N,)
    
    isolated_points = torch.where(neighbor_counts == 0)[0]
    print(f"Final result: {isolated_points.shape[0]} isolated points out of {distances.shape[0]} total")
    
    pts_index.reset()
    return isolated_points

def compute_new_points_ids_old(frustum_points: torch.Tensor, new_pts: torch.Tensor,
                           radius: float = 0.03, device: str = "cpu") -> torch.Tensor:
    """ Having newly initialized points, decides which of them should be added to the map.
        For every new point, if there are no neighbors within the radius in the frustum points,
        it is added to the submap.
    Args:
        frustum_points: Point within a current frustum of the active submap of shape (N, 3)
        new_pts: New 3D Gaussian means which are about to be added to the submap of shape (N, 3)
        radius: Radius whithin which the points are considered to be neighbors
        device: Execution device
    Returns:
        Indicies of the new points that should be added to the submap of shape (N)
    """
    if frustum_points.shape[0] == 0:
        return torch.arange(new_pts.shape[0])
    if device == "cpu":
        pts_index = faiss.IndexFlatL2(3)
    else:
        print(f"Using GPU for faiss index on device {device}")
        pts_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(3))
    
    # frustum_points = frustum_points.to(device)
    # new_pts = new_pts.to(device)
    
    if isinstance(frustum_points, torch.Tensor):
        frustum_points = frustum_points.detach().cpu().numpy()
        print(f"Shape of frustum_points: {frustum_points.shape}")
    
    frustum_points = np.ascontiguousarray(frustum_points.astype(np.float32))
    pts_index.add(frustum_points) # 将视锥内的点作为搜索数据库

    # 分批最近邻搜索
    split_pos = torch.split(new_pts, 65535, dim=0)
    distances, ids = [], []
    for split_p in split_pos:
        # 将 torch.Tensor 转换为 numpy 数组
        split_p_np = split_p.detach().cpu().numpy().astype(np.float32)
        split_p_np = np.ascontiguousarray(split_p_np)
        distance, id = pts_index.search(split_p_np, 8) # 搜索每个新点的在视锥内的8个最近邻
        # 打印平均距离
        print(f"avg distance to neighbors: {distance.mean():.4f}")
        distances.append(torch.from_numpy(distance).to(device))
        ids.append(torch.from_numpy(id).to(device))
    distances = torch.cat(distances, dim=0) # (新点数, 8) 新点到视锥内8个最近邻的距离
    ids = torch.cat(ids, dim=0) # (新点数, 8) 8个最近邻对应的ID
    neighbor_num = (distances < radius).sum(axis=1).int()
    pts_index.reset()
    # 返回没有邻居的点的索引
    return torch.where(neighbor_num == 0)[0] 

def compute_opt_views_distribution(keyframes_num, iterations_num, current_frame_iter) -> np.ndarray:
    """ Computes the prob
    ability distribution for selecting views based on the current iteration.
    Args:
        keyframes_num: The total number of keyframes.
        iterations_num: The total number of iterations planned.
        current_frame_iter: The current iteration number.
    Returns:
        An array representing the probability distribution of keyframes.
    """
    if keyframes_num == 1:
        return np.array([1.0])
    prob = np.full(keyframes_num, (iterations_num - current_frame_iter) / (keyframes_num - 1))
    prob[0] = current_frame_iter
    prob[1:] = prob[1:] * (1 - (current_frame_iter / iterations_num))
    prob /= prob.sum()
    return prob

def compute_density_map(pcds_world, w2c, K, image_size, point_radius=1.0, normalize=True):
    """
        使用GPU实现的密度图生成，每个点在其周围区域进行高斯加权 splatting。

        Args:
            pcds_cam: (N, 3) torch.Tensor 相机坐标系下的点
            K: (3, 3) 相机内参
            image_size: (H, W)
            point_radius: 高斯核标准差（单位：像素）
            normalize: 是否归一化输出

        Returns:
            density_map: (H, W) torch.Tensor
    """


    # visibility_mask = render_dict["visibility_filter"] # torch.bool
    # means3D = gaussian_model.get_xyz[visibility_mask] # 世界坐标系下的点
    # 齐次坐标转换
    pcds_world = torch.cat([pcds_world, torch.ones(pcds_world.shape[0], 1, device=pcds_world.device)], dim=1)  # (N, 4)
    pcds_cam = pcds_world @ w2c.T  # 转换
    pcds_cam = pcds_cam[:, :3]  # (N, 3)

    device = pcds_cam.device
    H, W = image_size

    # 筛除相机背后的点
    z = pcds_cam[:, 2]
    # print("Depth range:", z.min().item(), z.max().item())
    valid = z > 1e-5
    
    
    pcds_cam = pcds_cam[valid]
    z = z[valid]

    # 像素坐标 (u, v)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = pcds_cam[:, 0] / z
    y = pcds_cam[:, 1] / z
    u = fx * x + cx
    v = fy * y + cy

    # 滤除落在图像外远离区域的点，保留在图像内以及周围 margin 范围内的点
    margin = int(torch.ceil(torch.tensor(3 * point_radius)))
    mask = (u >= -margin) & (u < W + margin) & (v >= -margin) & (v < H + margin)
    u = u[mask]
    v = v[mask]

    N = u.shape[0]
    radius = int(torch.ceil(torch.tensor(3 * point_radius)))

    # 准备邻域偏移坐标 (dx, dy)，形状：(K, 2)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')  # shape: (K, K)
    offset = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (K², 2)
    K2 = offset.shape[0]

    # 展开点坐标并加偏移，生成所有影响的像素坐标 (N * K², 2)
    u_exp = u.view(-1, 1).expand(-1, K2)
    v_exp = v.view(-1, 1).expand(-1, K2)
    offset_u = offset[:, 0].view(1, -1)
    offset_v = offset[:, 1].view(1, -1)

    u_grid = u_exp + offset_u  # (N, K²)
    v_grid = v_exp + offset_v  # (N, K²)

    # 计算高斯权重
    du = u_grid - u.view(-1, 1)
    dv = v_grid - v.view(-1, 1)
    weight = torch.exp(-(du**2 + dv**2) / (2 * point_radius**2))  # (N, K²)

    # 过滤落在图像边界外的像素
    u_grid_clamped = u_grid.round().long()
    v_grid_clamped = v_grid.round().long()
    valid_mask = (u_grid_clamped >= 0) & (u_grid_clamped < W) & (v_grid_clamped >= 0) & (v_grid_clamped < H)

    # 构造 index 和 value（flat index 的方式）
    flat_idx = v_grid_clamped * W + u_grid_clamped  # (N, K²)
    flat_idx = flat_idx[valid_mask]
    value = weight[valid_mask]

    # 统计加权值并写入二维图像
    density_flat = torch.zeros(H * W, device=device)
    density_flat = density_flat.index_add(0, flat_idx, value)
    density_map = density_flat.view(H, W)

    if normalize and density_map.max() > 0:
        density_map = density_map / (density_map.max() + 1e-5)

    return density_map  # shape: (H, W), float32 on GPU

def compute_seeding_mask (args, background,gaussian_model, viewpoint: Camera, env_map: EnvLight,vis=True,frame_id=None,cam_id=None) -> np.ndarray:
    """
    Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded
    based on alpha masks or depth error.
    Args:
        gaussian_model: The current submap
        viewpoint: The camera viewpoint
    Returns:
        np.ndarray: A binary mask of shape (H, W) indicates regions suitable for seeding new 3D Gaussian models
    """
    with torch.no_grad():
        render_dict = render_func(viewpoint, gaussian_model, args, background, env_map=env_map, is_training=False)

        sky_mask  = viewpoint.sky_mask.cpu().numpy() # torch.bool
        semantic_mask = viewpoint.semantic_mask.detach().cpu().numpy() # torch.bool
        # Part1： 透明度过低的区域
        alpha = render_dict["alpha"].detach().cpu().numpy() 
        render_depth = render_dict["depth"].detach().cpu().numpy() 
        alpha_thre = np.percentile(alpha, 60) # 取透明度的10%作为阈值
        alpha_mask = (alpha < alpha_thre) & (~sky_mask) & (semantic_mask)  # torch.bool 将透明度最低的10%作为seeding区域


        # fill_depth_tensor = viewpoint.pts_depth # 稠密深度
        fill_depth_tensor = viewpoint.gt_pts_depth.detach().cpu().numpy() # 稀疏深度
        depth_error = np.abs(fill_depth_tensor -  render_depth) * (fill_depth_tensor > 0)  # 计算深度误差，忽略天空部分和零深度部分
        depth_thre = np.percentile(depth_error, 90) # 取深度误差的90%作为阈值
        
        # Part2： 深度比真值深度大且误差过大的区域
        depth_error_mask = ((render_depth> fill_depth_tensor) * (depth_error > depth_thre)) & (~sky_mask) & (semantic_mask)# torch.bool
        
        # seeding_mask = alpha_mask | depth_error_mask | edges_bool
        print("Depth error mask sum:", depth_error_mask.sum())
        print("Alpha mask sum:", alpha_mask.sum())
       
        seeding_mask = alpha_mask | depth_error_mask

    

    # 可视化部分
    # if vis:
        # render_rgb_to_vis = render_dict["render"].detach().cpu().numpy() #  (3, H, W)
        # render_rgb_to_vis = render_rgb_to_vis.transpose(1, 2, 0) # (H, W, 3)
        # render_rgb_to_vis = (render_rgb_to_vis * 255).astype(np.uint8)

        # gt_to_vis = viewpoint.original_image.cpu().numpy() # (3, H, W)
        # gt_to_vis = gt_to_vis.transpose(1, 2, 0) # (H, W, 3)
        # gt_to_vis = (gt_to_vis * 255).astype(np.uint8)

        # depth_error_to_vis = depth_error.squeeze()
        # depth_error_to_vis = (depth_error_to_vis * 255 / depth_error_to_vis.max()).astype(np.uint8) # 归一化到0-255
       
        # depth_error_to_vis = cv2.applyColorMap(depth_error_to_vis, cv2.COLORMAP_JET) # 使用jet colormap可视化深度误差
        # depth_error_to_vis = cv2.cvtColor(depth_error_to_vis, cv2.COLOR_BGR2RGB) # 转换为RGB格式

        # # density_map_to_vis = density_map.cpu().numpy().squeeze() # (H, W)
        # # density_map_to_vis = density_map_to_vis ** 0.5 # 开平方根增强对比度
        # # density_map_to_vis = (density_map_to_vis * 255).astype(np.uint8)
        # # density_map_to_vis = cv2.applyColorMap(density_map_to_vis, cv2.COLORMAP_JET)
        # # density_map_to_vis = cv2.cvtColor(density_map_to_vis, cv2.COLOR_BGR2RGB)

        # # 可视化seeding_mask
        # seeding_mask_to_vis = seeding_mask
        # seeding_mask_to_vis = seeding_mask_to_vis.squeeze().astype(np.uint8) * 255

        # # 将真实rgb、渲染rgb、seeding_mask绘制在一张图上
        # fig,ax = plt.subplots(1, 6, figsize=(20, 5))
        # ax[0].imshow(gt_to_vis)
        # ax[0].set_title("GT RGB")
        # ax[1].imshow(render_rgb_to_vis)
        # ax[1].set_title("Render RGB")
        # ax[2].imshow(depth_error_to_vis)
        # ax[2].set_title("Depth Error")
        # # ax[3].imshow(density_map_to_vis)
        # # ax[3].set_title("Density Map")
        # ax[3].imshow(seeding_mask_to_vis, cmap='gray')
        # ax[3].set_title(f"Seeding Mask {seeding_mask.sum()} ")

        # ax[4].imshow(alpha_mask.squeeze(), cmap='gray')
        # ax[4].set_title(f"Alpha Mask {alpha_mask.sum()} ")

        # ax[5].imshow(depth_error_mask.squeeze(), cmap='gray')
        # ax[5].set_title(f"Depth Error Mask {depth_error_mask.sum()} ")

        # plt.tight_layout()
        # os.makedirs(os.path.join(args.model_path, "seeding_mask"), exist_ok=True)
        # output_path = os.path.join(args.model_path, "seeding_mask", f"{frame_id}_{cam_id}.png")
        # plt.savefig(output_path)

    return seeding_mask



def create_point_cloud(image: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Creates a point cloud from an image, depth map, camera intrinsics, and pose.

    Args:
        image: The RGB image of shape (H, W, 3)
        depth: The depth map of shape (H, W)
        intrinsics: The camera intrinsic parameters of shape (3, 3)
        pose: c2w matrix of shape (4, 4)
    Returns:
        A point cloud of shape (N, 6) with last dimension representing (x, y, z, r, g, b)
    """
    height, width = depth.shape
    # Create a mesh grid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    # Convert pixel coordinates to camera coordinates
    x = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z = depth
    # Stack the coordinates together
    points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
    # Reshape the coordinates for matrix multiplication
    points = points.reshape(-1, 4)
    # Transform points to world coordinates
    posed_points = pose @ points.T
    posed_points = posed_points.T[:, :3] # (N, 3)
    # Flatten the image to get colors for each point
    colors = image.reshape(-1, 3)
    # Concatenate posed points with their corresponding color
    point_cloud = np.concatenate((posed_points, colors), axis=-1)

    return point_cloud



def back_project_from_depth(args, viewpoint: Camera, seeding_mask: np.ndarray, frame_id: int,cam_id: int) ->BasicPointCloud:
        """Back-project RGB-D pixels inside the seeding mask into world coordinates.

        The function uses sparse LiDAR depth, RGB colors, intrinsics, and the camera pose
        to reconstruct a dense RGB point cloud. Pixels marked by `seeding_mask` and with
        valid depth (non-zero, non-sky) are retained, projected to 3D space, and converted
        into a `BasicPointCloud` that serves as candidate Gaussians for dynamic refinement.

        Args:
            args: Global hydra configuration (only used for visualization flags).
            viewpoint: Camera wrapper containing RGB, depth, intrinsics, pose, and masks.
            seeding_mask: Binary mask (H, W) selecting pixels to back-project.
            frame_id: Current frame index, used only for visualization bookkeeping.
            cam_id: Camera index within the frame, also used for visualization bookkeeping.

        Returns:
            BasicPointCloud: World-space XYZ plus RGB of all accepted pixels.
        """

        gt_color = viewpoint.original_image
        gt_color = gt_color.permute(1, 2, 0).cpu().numpy()  # Convert to HWC layout

        gt_depth = viewpoint.gt_pts_depth 
        gt_depth = gt_depth.cpu().numpy()  # 1 x H x W tensor -> numpy
        gt_depth = gt_depth.squeeze(0)  # Remove channel to get H x W map
        gt_depth = gt_depth.astype(np.float32)


        c2w = viewpoint.c2w.cpu().numpy()  # 4x4
        K = viewpoint.get_k().cpu().numpy()  # 3x3
        
        # Step 1: lift every RGB-D pixel into world coordinates
        pts = create_point_cloud(gt_color, gt_depth, K, c2w) # (N, 6) = xyz + rgb
        pts_xyz = pts[:, :3]  # (N, 3)

        flat_gt_depth = gt_depth.flatten() # (H*W, )
        flat_sky_mask = torch.where(viewpoint.sky_mask > 0, True, False).cpu().numpy().flatten()

        non_zero_depth_mask = flat_gt_depth > 0.
        non_zero_depth_mask = non_zero_depth_mask & (~flat_sky_mask)  # Drop sky pixels and zero depth
        valid_ids = np.flatnonzero(seeding_mask)  # Indices selected by seeding mask
        valid_ids = valid_ids[non_zero_depth_mask[valid_ids]]  # Keep only valid depths
        pts = pts[valid_ids, :].astype(np.float32)
        cloud_to_add = pts.astype(np.float32)

        pts_xyz = cloud_to_add[:, :3] # (N, 3)
        pts_rgb = cloud_to_add[:, 3:] # (N, 3)

        visualize_new_points_projection(args, viewpoint, pts_xyz, frame_id, cam_id)
        pcd = BasicPointCloud( pts_xyz , colors = pts_rgb, normals=None)

        return pcd



def seed_new_point_cloud(args, viewpoint: Camera, exist_pcd: torch.tensor, seeding_mask: np.ndarray, frame_id: int,cam_id: int,init=True) ->BasicPointCloud:
        """
        Seeds means for the new 3D Gaussian based on ground truth color and depth, camera intrinsics,
        estimated camera-to-world transformation, a seeding mask, and a flag indicating whether this is a new submap.
        Args:
            gt_color: The ground truth color image as a numpy array with shape (H, W, 3).
            gt_depth: The ground truth depth map as a numpy array with shape (H, W).
            intrinsics: The camera intrinsics matrix as a numpy array with shape (3, 3).
            estimate_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            init: A boolean flag indicating whether this is the initialization
        Returns:
            BasicPointCloud: A point cloud object containing the seeded points.
        """

        gt_color = viewpoint.original_image
        gt_color = gt_color.permute(1, 2, 0).cpu().numpy()  # HWC

        # gt_depth = viewpoint.pts_depth # 稠密点云深度
        gt_depth = viewpoint.gt_pts_depth # 稀疏点云深度
        gt_depth = gt_depth.cpu().numpy()  # 1xHxW
        gt_depth = gt_depth.squeeze(0)  # HxW
        gt_depth = gt_depth.astype(np.float32)


        c2w = viewpoint.c2w.cpu().numpy()  # 4x4
        K = viewpoint.get_k().cpu().numpy()  # 3x3
        
        # Step 1: 从RGBD创建整体点云，一个像素对应一个点
        pts = create_point_cloud(gt_color, gt_depth, K, c2w) # (N, 6) 前三列是xyz，后三列是rgb
        pts_xyz = pts[:, :3]  # (N, 3)

        flat_gt_depth = gt_depth.flatten() # (H*W, )
        flat_sky_mask = torch.where(viewpoint.sky_mask > 0, True, False).cpu().numpy().flatten()

        non_zero_depth_mask = flat_gt_depth > 0.
        non_zero_depth_mask = non_zero_depth_mask & (~flat_sky_mask)# 过滤掉天空部分和零深度部分 (H*W, )
        valid_ids = np.flatnonzero(seeding_mask) # flatnonzero返回非零元素的索引 
        valid_ids = valid_ids[non_zero_depth_mask[valid_ids]] # 过滤掉天空部分和零深度部分 (N, )
        pts = pts[valid_ids, :].astype(np.float32)


        # #需要的阈值

        # # init_unit_sample_size: 50000 # 初始化全图均匀采样GS点的数量
        # # init_gradient_sample_size: 50000 # 初始化按照梯度采样GS点的数量
        # # unit_sample_size: 10000 # 后续全图进行均匀采样GS点的数量
        # # gradient_sample_size: 10000 # 后续根据sobel梯度采样的点
        # # seeding_mask_sample_size: 5000 # 后续根据seeding mask采样的点


        
        # # Step 2: 采样
        # if init:
        #     # Part1 : 全图均匀采样
        #     if args.init_unit_sample_size < 0:
        #         uniform_ids = np.arange(pts.shape[0])
        #     else:
        #         # 从所有点中随机选取init_unit_sample_size个不同的点 （当前分辨率为610,000 最多采样610,000个点）
        #         uniform_ids = np.random.choice(pts.shape[0], args.init_unit_sample_size, replace=False)
            
        #     # Part2 : 基于RGB梯度采样
        #     gradient_ids = sample_pixels_based_on_gradient(gt_color, args.init_gradient_sample_size)
            
        #     # Part3 : seeding mask采样(seeding_mask初始化时只提供Canny边缘，需要全部seeding)
        #     combined_ids = np.concatenate((uniform_ids, gradient_ids))
        #     combined_ids = np.concatenate((combined_ids, valid_ids))
        #     # 初始化采样： seeding mask + 全图均匀采样 + 基于RGB梯度采样 + 去重
        #     sample_ids = np.unique(combined_ids)
        # else:
        #     # Part1 : 全图均匀采样
        #     if args.unit_sample_size < 0:
        #         uniform_ids = np.arange(pts.shape[0])
        #     else:
        #         uniform_ids = np.random.choice(pts.shape[0], args.unit_sample_size, replace=False)

        #     # Part2 : 基于RGB梯度采样
        #     gradient_ids = sample_pixels_based_on_gradient(gt_color, args.gradient_sample_size)

        #     # Part3 : seeding mask采样 (后续帧Seeding mask包含透明度过低+深度空洞)
        #     if args.seeding_mask_sample_size < 0:
        #         seeding_mask_ids = valid_ids
        #     else:
        #         if len(valid_ids)==0:
        #             print("Warning: No valid ids in seeding mask, using empty array for seeding_mask_ids")
        #             seeding_mask_ids = np.array([], dtype=np.int64)
        #         else:
        #             seeding_mask_ids = np.random.choice(valid_ids, size=args.seeding_mask_sample_size, replace=True)

        #     combined_ids = np.concatenate((uniform_ids, gradient_ids, seeding_mask_ids))
        #     sample_ids = np.unique(combined_ids) # 去重
            
            
        
        # sample_ids = sample_ids[non_zero_depth_mask[sample_ids]]
        # pts = pts[sample_ids, :].astype(np.float32)
       
        # # Step3：计算视锥体内已有点的索引
        # if not init:
        #     gaussian_points = exist_pcd.to("cuda") # (N, 3) torch.Tensor
        #     camera_frustum_corners = compute_camera_frustum_corners(gt_depth, c2w,K)
        #     # (8,3) 视锥体在世界坐标系下的八个角点
        #     camera_frustum_corners = torch.from_numpy(camera_frustum_corners).to("cuda") # np.ndarray to torch.Tensor
        #     reused_pts_ids = compute_frustum_point_ids(gaussian_points, camera_frustum_corners, device="cuda")
        #     # print(f"Reused points in frustum: {reused_pts_ids.shape[0]} from {gaussian_points.shape[0]} total points")
        #     # Step4: 计算在视锥中孤立的新点的索引
        #     new_pts_ids = compute_new_points_ids(gaussian_points[reused_pts_ids], torch.from_numpy(pts[:, :3]).to("cuda").contiguous(),
        #                                         radius=args.new_points_radius, device="cuda")
        #     new_pts_ids = new_pts_ids.cpu().numpy() # torch.Tensor to np.ndarray
        #     print(f"Seeding new points: {new_pts_ids.shape[0]} with {pts.shape[0]- new_pts_ids.shape[0]} points already in the frustum")
            
        #     if new_pts_ids.shape[0] > 0:
        #         cloud_to_add = pts[new_pts_ids, :].astype(np.float32) # (N, 6)
        #     else:
        #         cloud_to_add = np.empty((0, 6), dtype=np.float32)
        # else:
        #     # 初始化时，直接使用采样的点
        cloud_to_add = pts.astype(np.float32)

        pts_xyz = cloud_to_add[:, :3] # (N, 3)
        pts_rgb = cloud_to_add[:, 3:] # (N, 3) # (N, 1)
        # print(f"frame {frame_id} cam {cam_id} seed new point cloud with {pts_xyz.shape[0]} points")

        visualize_new_points_projection(args, viewpoint, pts_xyz, frame_id, cam_id)

        # ps：在原代码中，初始化完整点云场的时候，颜色采用的全是黑色
        pcd = BasicPointCloud( pts_xyz , colors = pts_rgb, normals=None)

        return pcd

def collect_pcd(pcd_list: list):
    
    
    pcd_init = BasicPointCloud(
                points = np.concatenate([pcd.points for pcd in pcd_list], axis=0),
                colors = np.concatenate([pcd.colors for pcd in pcd_list], axis=0),
                normals = np.concatenate([pcd.normals for pcd in pcd_list], axis=0) if pcd_list[0].normals is not None else None,
                time = np.concatenate([pcd.time for pcd in pcd_list], axis=0) if pcd_list[0].time is not None else None
            )
    
    return pcd_init

def plt_points_change(num_list,output_path,frame_id):
    """
    Plots the number of points in each iteration and saves the plot to the specified path.
    Args:
        num_list: A list of integers representing the number of points in each iteration.
        output_path: The path where the plot will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(num_list, marker='o')
    plt.title('Number of Points in Each Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Points')
    plt.grid()
    os.makedirs(os.path.join(output_path,"pts_num_vis"), exist_ok=True)
    output_path_fig = os.path.join(output_path,"pts_num_vis", f"points_change_{frame_id}.png")
    plt.savefig(output_path_fig)
    plt.close()


def visualize_new_points_projection(args, viewpoint: Camera, new_points: np.ndarray, frame_id: int, cam_id: int):
    """
    可视化新添加的点投影到像素平面
    Args:
        args: 参数配置
        viewpoint: 相机视点
        new_points: 新添加的3D点 (N, 3)
        frame_id: 帧ID
        cam_id: 相机ID
    """
    if new_points.shape[0] == 0:
        print("No new points to visualize")
        return
    
    # 获取相机参数
    K = viewpoint.get_k().cpu().numpy()  # (3, 3)
    c2w = viewpoint.c2w.cpu().numpy()  # (4, 4)
    w2c = np.linalg.inv(c2w)  
    w2c = viewpoint.world_view_transform.cpu().numpy().transpose(   )  # (4, 4) 世界到相机的变换矩阵
    
    # 将世界坐标点转换为相机坐标
    new_points_homo = np.concatenate([new_points, np.ones((new_points.shape[0], 1))], axis=1)  # (N, 4)
    points_cam = (w2c @ new_points_homo.T).T  # (N, 4)
    points_cam = points_cam[:, :3]  # (N, 3)
    z = points_cam[:,2]
    
    # 过滤掉相机后面的点
    valid_depth = points_cam[:, 2] > 1e-5 
    points_cam = points_cam[valid_depth]
    
    if points_cam.shape[0] == 0:
        print("No valid points in front of camera")
        return
    
    # 投影到像素平面
    points_2d = points_cam @ K.T  # (N, 3)
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # (N, 2) 齐次坐标归一化
    
    # 过滤图像边界内的点
    H, W = viewpoint.image_height, viewpoint.image_width
    valid_proj = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & \
                 (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
    points_2d = points_2d[valid_proj]
    points_depths = points_cam[valid_proj, 2]  # 对应的深度值
    
    
    # 获取原始图像
    gt_image = viewpoint.original_image.cpu().numpy()  # (3, H, W)
    gt_image = gt_image.transpose(1, 2, 0)  # (H, W, 3)
    gt_image_vis = (gt_image * 255).astype(np.uint8)
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 第一个子图：原始图像 + 新点投影（按深度着色）
    axes[0].imshow(gt_image_vis)
    if points_2d.shape[0] > 0:
        # 根据深度值着色
        scatter = axes[0].scatter(points_2d[:, 0], points_2d[:, 1], 
                                 c=points_depths, cmap='plasma', s=2, alpha=0.8)
        plt.colorbar(scatter, ax=axes[0], label='Depth (m)')
    axes[0].set_title(f'New Points Projection (Depth Colored)\nTotal: {points_2d.shape[0]} points')
    axes[0].set_xlim(0, W)
    axes[0].set_ylim(H, 0)  # 翻转Y轴
    
    # 第二个子图：密度热力图
    density_map = np.zeros((H, W), dtype=np.float32)
    if points_2d.shape[0] > 0:
        # 统计每个像素的点数量
        u_int = np.clip(points_2d[:, 0].astype(int), 0, W-1)
        v_int = np.clip(points_2d[:, 1].astype(int), 0, H-1)
        
        for u, v in zip(u_int, v_int):
            density_map[v, u] += 1
        
        # 高斯模糊平滑密度图
        density_map = gaussian_filter(density_map, sigma=2.0)
    
    im = axes[1].imshow(density_map, cmap='hot', alpha=0.8)
    axes[1].imshow(gt_image_vis, alpha=0.3)  # 叠加原图
    plt.colorbar(im, ax=axes[1], label='Point Density')
    axes[1].set_title('New Points Density Map')
    
    # 第三个子图：分区域统计
    axes[2].imshow(gt_image_vis)
    if points_2d.shape[0] > 0:
        # 将图像分成4x4网格，统计每个区域的点数
        grid_h, grid_w = 4, 4
        cell_h, cell_w = H // grid_h, W // grid_w
        
        for i in range(grid_h):
            for j in range(grid_w):
                # 计算网格边界
                y1, y2 = i * cell_h, (i + 1) * cell_h if i < grid_h - 1 else H
                x1, x2 = j * cell_w, (j + 1) * cell_w if j < grid_w - 1 else W
                
                # 统计该区域内的点数
                in_cell = (points_2d[:, 0] >= x1) & (points_2d[:, 0] < x2) & \
                         (points_2d[:, 1] >= y1) & (points_2d[:, 1] < y2)
                count = in_cell.sum()
                
                # 绘制网格和点数
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='yellow', linewidth=2)
                axes[2].add_patch(rect)
                
                # 在网格中心显示点数
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                axes[2].text(cx, cy, str(count), color='yellow', fontsize=12, 
                           ha='center', va='center', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    axes[2].set_title('Point Distribution by Regions')
    axes[2].set_xlim(0, W)
    axes[2].set_ylim(H, 0)
    
    plt.tight_layout()
    
    # 保存可视化结果
    os.makedirs(os.path.join(args.model_path, "new_points_projection"), exist_ok=True)
    output_path = os.path.join(args.model_path, "new_points_projection", f"frame_{frame_id}_cam_{cam_id}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # print(f"New points projection visualization saved to: {output_path}")
    
    # # 打印统计信息
    # print(f"Statistics for new points:")
    # print(f"  Total new points in 3D: {new_points.shape[0]}")
    # print(f"  Points in front of camera: {points_cam.shape[0] if 'points_cam' in locals() else 0}")
    # print(f"  Points projected to image: {points_2d.shape[0]}")
    # if points_2d.shape[0] > 0:
    #     print(f"  Depth range: [{points_depths.min():.2f}, {points_depths.max():.2f}] m")
    #     print(f"  Average depth: {points_depths.mean():.2f} m")
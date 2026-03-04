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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
from kornia.filters import laplacian, spatial_gradient
import numpy as np
import faiss
import time
def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map
    
def tv_loss(depth):
    c, h, w = depth.shape[0], depth.shape[1], depth.shape[2]
    count_h = c * (h - 1) * w
    count_w = c * h * (w - 1)
    h_tv = torch.square(depth[..., 1:, :] - depth[..., :h-1, :]).sum()
    w_tv = torch.square(depth[..., :, 1:] - depth[..., :, :w-1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask


def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def cal_gradient(data):
    """
    data: [1, C, H, W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    gradient = torch.abs(grad_x) + torch.abs(grad_y)

    return gradient


def bilateral_smooth_loss(data, image, mask):
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    rgb_grad = cal_gradient(image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]
    data_grad = cal_gradient(data.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]

    smooth_loss = (data_grad * (-rgb_grad).exp() * mask).mean()

    return smooth_loss


def second_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs() * torch.exp(-10*spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()


def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def first_order_edge_aware_norm_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].norm(dim=1, keepdim=True))).sum(1).mean()

def first_order_loss(data):
    return spatial_gradient(data[None], order=1)[0].abs().sum(1).mean()

def loss_cls_3d_dynamic_static(positions, predictions, mask_A, mask_B, k=5, batch_size=200000,detach_dynamic_prob=False):
    """
    计算动态点云A与静态点云B之间的3D分类一致性损失
    
    Args:
        positions: 所有点的空间位置，形状为 (N, 3)
        predictions: 所有点的预测概率向量，形状为 (N, C)
        mask_A: 指示A点集的布尔掩码，形状为 (N,)
        mask_B: 指示B点集的布尔掩码，形状为 (N,)
        k: 要找到A点云在B点云中的k近邻，默认值为5
        batch_size: 分批处理B点云的批大小，默认200000
    
    Returns:
        normalized_loss: 归一化的3D KL散度损失 [0, 1]
        knn_mask: 形状为(N,)的布尔掩码，指示B中被选为A的k近邻的点
    """
    device = positions.device
    
    # 提取A和B点云
    positions_A = positions[mask_A]  # (N_A, 3)
    if detach_dynamic_prob:
        predictions_A = predictions[mask_A].detach()  # (N_A, C)
    else:  
        predictions_A = predictions[mask_A]  # (N_A, C)
    positions_B = positions[mask_B]  # (N_B, 3)
    predictions_B = predictions[mask_B]  # (N_B, C)
    
    N_A = positions_A.shape[0]
    N_B = positions_B.shape[0]
    
    # 创建一个新的掩码来标记B中的近邻点
    knn_mask = torch.zeros_like(mask_B, dtype=torch.bool, device=device)
    
    # 如果A点云为空，返回0损失和空掩码
    if N_A == 0:
        return torch.tensor(0.0, device=device), knn_mask
    

    # start_time = time.time()
    # 转换为numpy数组进行FAISS搜索
    pos_A_np = positions_A.detach().cpu().numpy()
    pos_B_np = positions_B.detach().cpu().numpy()
    
    # 计算所有点的中心和缩放因子，用于归一化
    all_points = np.vstack([pos_A_np, pos_B_np])
    center = np.mean(all_points, axis=0)  # (3,)
    scale = np.max(np.abs(all_points - center)) + 1e-6
    
    # 归一化坐标到[-1, 1]范围
    pos_A_norm = (pos_A_np - center) / scale  # (N_A, 3)
    pos_B_norm = (pos_B_np - center) / scale  # (N_B, 3)
    
    # 准备A点云数据用于搜索
    pos_A_norm = np.ascontiguousarray(pos_A_norm.astype(np.float32))
    
    # 分批处理B点云，收集所有距离和索引
    all_distances = []
    all_indices = []
    
    # 对B点云进行分批处理
    num_B_batches = (N_B + batch_size - 1) // batch_size
    
    for i in range(num_B_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N_B)
        
        # 当前B点云批次
        batch_pos_B = pos_B_norm[start_idx:end_idx]  # (batch_size, 3)
        batch_pos_B = np.ascontiguousarray(batch_pos_B.astype(np.float32))
        
        # 为当前B批次创建FAISS索引
        if device.type == "cpu":
            pts_index = faiss.IndexFlatL2(3)
        else:
            # print("Using FAISS on GPU")
            pts_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(3))
        
        # 将当前B批次添加到索引中
        pts_index.add(batch_pos_B)
        
        # 使用所有A点云搜索当前B批次中的k个最近邻
        distances, local_ids = pts_index.search(pos_A_norm, k)  # (N_A, k)
        
        # 去除无效的距离
        distances = np.maximum(distances, 0.0)  # 确保非负
        distances = np.nan_to_num(distances, nan=1e10, posinf=1e10, neginf=0.0)  # 处理NaN和无穷大
        
        # 转换回欧几里得距离
        distances_euclidean = np.sqrt(distances) * scale # (N_A, k)
        
        # 将局部索引转换为全局索引
        global_ids = local_ids + start_idx # (N_A, k)
        
        # 转换为torch张量并移动到正确设备
        distances_batch = torch.from_numpy(distances_euclidean).to(device)
        indices_batch = torch.from_numpy(global_ids).to(device)
        
        all_distances.append(distances_batch)
        all_indices.append(indices_batch)
    
    # 合并所有批次的距离和索引 (N_A, num_batches*k)
    full_distances = torch.cat(all_distances, dim=1)  # (N_A, total_neighbors)
    full_indices = torch.cat(all_indices, dim=1)      # (N_A, total_neighbors)
    
    # 为每个A点找到真正的k个最近邻
    topk_distances, topk_positions = full_distances.topk(k, dim=1, largest=False)  # (N_A, k)
    
    # 使用位置索引获取真正的B点云索引
    neighbor_indices = torch.gather(full_indices, 1, topk_positions)  # (N_A, k)

    # print("FAISS search time:", time.time() - start_time)
    
    # 获取B点中原始索引，用于构建KNN掩码
    B_original_indices = torch.where(mask_B)[0]  # B中点在原始点云中的索引
    
    # 构建KNN掩码
    neighbor_global_indices = B_original_indices[neighbor_indices.flatten()]  # (N_A*k,)
    knn_mask[neighbor_global_indices] = True

    # 获取A点云中每个点的k近邻在B点云中的预测概率
    neighbor_preds = predictions_B[neighbor_indices]  # (N_A, k, C)
    
    # 计算KL散度损失
    # 将A点云的预测概率扩展维度以便与邻居预测进行比较
    sample_preds_expanded = predictions_A.unsqueeze(1)  # (N_A, 1, C)
    
    # 计算KL散度：KL(P_A || P_B)
    kl_divergence = sample_preds_expanded * (
        torch.log(sample_preds_expanded + 1e-10) -
        torch.log(neighbor_preds + 1e-10)
    )  # (N_A, k, C)

    # 对类别维度求和，然后对邻居维度求平均，最后对所有A点求平均
    total_loss = kl_divergence.sum(dim=-1).mean()  # 对C维度求和，对N_A和k维度求均值
    
    # 归一化损失到[0, 1]范围
    num_classes = predictions_A.size(1)
    normalized_loss = total_loss / num_classes
    
    return normalized_loss, knn_mask

def loss_velocity_consistency(positions, velocities, beta, k=5, alpha=0.4, gamma=0.3):
    """
    计算点云内部的速度一致性损失，同时考虑速度大小、方向和beta标量的相似性
    
    Args:
        positions: 点云的空间位置，形状为 (N, 3)
        velocities: 点云的速度向量，形状为 (N, 3)
        beta: 点云的beta标量属性，形状为 (N,) 或 (N, 1)
        k: 每个点考虑的近邻数量，默认值为5
        alpha: 大小损失的权重，默认0.5
        gamma: beta损失的权重，(1-alpha-gamma)为方向损失的权重，默认0.3
    
    Returns:
        combined_loss: 结合大小、方向和beta的一致性损失
    """
    device = positions.device
    N = positions.shape[0]
    
    # 如果点云为空，返回0损失
    if N == 0:
        return torch.tensor(0.0, device=device)
    
    # 确保beta是正确的形状
    if beta.dim() == 2 and beta.shape[1] == 1:
        beta = beta.squeeze(1)  # (N, 1) -> (N,)
    
    # 转换为numpy数组进行FAISS搜索
    pos_np = positions.detach().cpu().numpy()
    
    # 计算所有点的中心和缩放因子，用于归一化
    center = np.mean(pos_np, axis=0)  # (3,)
    scale = np.max(np.abs(pos_np - center)) + 1e-6
    
    # 归一化坐标到[-1, 1]范围
    pos_norm = (pos_np - center) / scale  # (N, 3)
    
    # 准备数据用于FAISS搜索
    pos_norm = np.ascontiguousarray(pos_norm.astype(np.float32))
    
    # 创建FAISS索引
    if device.type == "cpu":
        pts_index = faiss.IndexFlatL2(3)
    else:
        pts_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(3))
    
    # 将所有点云添加到索引中
    pts_index.add(pos_norm)
    
    # 为所有点搜索k个最近邻
    distances, neighbor_indices = pts_index.search(pos_norm, k)  # (N, k)
    
    # 转换为torch张量并移动到正确设备
    neighbor_indices = torch.from_numpy(neighbor_indices).to(device)  # (N, k)
    
    # 获取每个点的k近邻的速度向量和beta值
    neighbor_velocities = velocities[neighbor_indices]  # (N, k, 3)
    neighbor_betas = beta[neighbor_indices]  # (N, k)
    velocities_expanded = velocities.unsqueeze(1)  # (N, 1, 3)
    betas_expanded = beta.unsqueeze(1)  # (N, 1)
    
    # 1. 速度大小一致性损失 (L2距离)
    velocity_diff = neighbor_velocities - velocities_expanded  # (N, k, 3)
    magnitude_loss = torch.norm(velocity_diff, dim=2).mean()  # (N, k) -> scalar
    
    # 2. 速度方向一致性损失 (余弦相似度)
    # 归一化速度向量
    vel_norm = F.normalize(velocities_expanded, dim=2, eps=1e-8)  # (N, 1, 3)
    neighbor_vel_norm = F.normalize(neighbor_velocities, dim=2, eps=1e-8)  # (N, k, 3)
    
    # 计算余弦相似度
    cosine_sim = (vel_norm * neighbor_vel_norm).sum(dim=2)  # (N, k)
    
    # 余弦损失：1 - cosine_similarity，值域[0, 2]
    direction_loss = (1 - cosine_sim).mean()  # scalar
    
    # 3. Beta一致性损失 (L1距离)
    beta_diff = torch.abs(neighbor_betas - betas_expanded)  # (N, k)
    beta_loss = beta_diff.mean()  # scalar
    
    # 4. 归一化损失
    # 归一化大小损失
    velocity_scale = velocities.norm(dim=1).mean() + 1e-6
    normalized_magnitude_loss = magnitude_loss / velocity_scale
    
    # 方向损失已经在[0, 2]范围内，归一化到[0, 1]
    normalized_direction_loss = direction_loss / 2.0
    
    # 归一化beta损失
    beta_scale = beta.abs().mean() + 1e-6
    normalized_beta_loss = beta_loss / beta_scale
    
    # 5. 结合三种损失
    # 确保权重总和为1
    remaining_weight = 1.0 - alpha - gamma
    if remaining_weight < 0:
        # 如果权重设置不合理，自动调整
        total_weight = alpha + gamma
        alpha = alpha / total_weight * 0.7
        gamma = gamma / total_weight * 0.7
        remaining_weight = 0.3
    
    combined_loss = (alpha * normalized_magnitude_loss + 
                    remaining_weight * normalized_direction_loss + 
                    gamma * normalized_beta_loss)
    
    return combined_loss

def loss_velocity_consistency_fast(velocities, beta, alpha=0.4, gamma=0.3):
    """
    直接计算同一ID内所有点云的速度一致性损失，同时考虑速度大小、方向和beta标量的相似性
    针对点数较少的情况优化，计算所有点之间的一致性
    
    Args:
        velocities: 同一ID内所有点的速度向量，形状为 (N, 3)
        beta: 同一ID内所有点的beta标量属性，形状为 (N,) 或 (N, 1)
        alpha: 速度大小损失的权重，默认0.4
        gamma: beta损失的权重，默认0.3，剩余权重给方向损失
    
    Returns:
        combined_loss: 结合大小、方向和beta的一致性损失
    """
    device = velocities.device
    N = velocities.shape[0]
    
    # 如果点云数量少于2个，无法计算一致性
    if N < 2:
        return torch.tensor(0.0, device=device)
    
    # 确保beta是正确的形状
    if beta.dim() == 2 and beta.shape[1] == 1:
        beta = beta.squeeze(1)  # (N, 1) -> (N,)
    
    # 计算所有点与均值的差异（全局一致性）
    # 1. 速度大小一致性损失
    velocity_mean = velocities.mean(dim=0, keepdim=True)  # (1, 3)
    velocity_diff = velocities - velocity_mean  # (N, 3)
    magnitude_loss = torch.norm(velocity_diff, dim=1).mean()  # 平均L2距离
    
    # 2. 速度方向一致性损失
    # 计算所有速度向量的平均方向
    velocity_mean_norm = F.normalize(velocity_mean, dim=1, eps=1e-8)  # (1, 3)
    velocities_norm = F.normalize(velocities, dim=1, eps=1e-8)  # (N, 3)
    
    # 计算每个速度与平均方向的余弦相似度
    cosine_sim = (velocities_norm * velocity_mean_norm).sum(dim=1)  # (N,)
    direction_loss = (1 - cosine_sim).mean()  # 平均方向损失
    
    # 3. Beta一致性损失
    beta_mean = beta.mean()  # scalar
    beta_diff = torch.abs(beta - beta_mean)  # (N,)
    beta_loss = beta_diff.mean()  # 平均beta差异
    
    # 4. 归一化损失
    # 归一化速度大小损失
    velocity_scale = velocities.norm(dim=1).mean() + 1e-6
    normalized_magnitude_loss = magnitude_loss / velocity_scale
    
    # 方向损失已经在[0, 2]范围内，归一化到[0, 1]
    normalized_direction_loss = direction_loss / 2.0
    
    # 归一化beta损失
    beta_scale = beta.abs().mean() + 1e-6
    normalized_beta_loss = beta_loss / beta_scale
    
    # 5. 结合三种损失
    # 确保权重总和为1
    remaining_weight = 1.0 - alpha - gamma
    if remaining_weight < 0:
        # 如果权重设置不合理，自动调整
        total_weight = alpha + gamma
        alpha = alpha / total_weight * 0.7
        gamma = gamma / total_weight * 0.7
        remaining_weight = 0.3
    
    combined_loss = (alpha * normalized_magnitude_loss + 
                    remaining_weight * normalized_direction_loss + 
                    gamma * normalized_beta_loss)
    
    return combined_loss
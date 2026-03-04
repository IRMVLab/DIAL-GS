import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser

from scene import GaussianModel, EnvLight
from gaussian_renderer import get_renderer
from scene.waymo_increment_loader import WaymoDataset


def visualize_low_opacity_regions(args, dataset, gaussians, env_map, render_func, background):
    """
    渲染每一帧并可视化除去天空外透明度最低的50%区域
    """
    output_dir = os.path.join(args.model_path, "low_opacity_regions")
    os.makedirs(output_dir, exist_ok=True)
    
    print("可视化透明度最低的50%区域...")
    
    for frame_id in tqdm(range(dataset.frame_num)):
        for cam_id in range(dataset.cam_num):
            # 获取当前相机
            viewpoint_cam = dataset.get_cam(dataset.resolution_scales[0], frame_id, cam_id)
            
            # 渲染当前视角
            with torch.no_grad():
                render_pkg = render_func(
                    viewpoint_cam, gaussians, args, background,
                    env_map=env_map, is_training=False
                )
            
            # 获取透明度和天空掩码
            alpha = render_pkg["alpha"].detach().cpu().numpy().squeeze()  # (H, W)
            sky_mask = viewpoint_cam.sky_mask.cpu().numpy().squeeze() if viewpoint_cam.sky_mask is not None else np.zeros_like(alpha, dtype=bool)
            
            # 创建有效区域掩码 (非天空区域)
            valid_mask = ~sky_mask
            
            # 如果没有有效区域，跳过
            if not np.any(valid_mask):
                continue
            
            # 获取非天空区域的透明度值
            valid_alpha = alpha[valid_mask]
            
            # 计算透明度阈值
            threshold = np.percentile(valid_alpha, 20) # 10表示最低10%的透明度
            
            # 创建低透明度区域掩码（透明度低于阈值的区域为1，其余为0）
            low_opacity_mask = np.zeros_like(alpha)
            print(f"Frame {frame_id}, Cam {cam_id}: Valid Alpha Min: {valid_alpha.min()}, Max: {valid_alpha.max()}, Threshold: {threshold}")
            low_opacity_mask[valid_mask] = (valid_alpha < threshold).astype(np.float32)
            print(f"Frame {frame_id}, Cam {cam_id}: Low Opacity Regions: {np.sum(low_opacity_mask)} pixels")
            
            # 获取原始RGB图像用于叠加显示
            gt_rgb = viewpoint_cam.original_image.cpu().numpy().transpose(1, 2, 0)
            
            # 创建可视化图像
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 原始RGB图像
            axes[0].imshow(gt_rgb)
            axes[0].set_title("GT RGB")
            axes[0].axis("off")
            
            # 透明度图像
            opacity_vis = axes[1].imshow(alpha, cmap='viridis')
            axes[1].set_title("Opacity Map")
            axes[1].axis("off")
            plt.colorbar(opacity_vis, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 低透明度区域掩码（白色区域表示透明度低的区域）
            axes[2].imshow(low_opacity_mask, cmap='gray')
            axes[2].set_title("Low Opacity Mask (Lowest 50%)")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"frame_{frame_id}_cam_{cam_id}.png"))
            plt.close(fig)
            
            # 额外保存一个叠加在原始图像上的掩码图
            overlay = np.zeros((*low_opacity_mask.shape, 3))
            overlay[..., 0] = low_opacity_mask  # 红色通道
            
            plt.figure(figsize=(10, 6))
            plt.imshow(gt_rgb)
            plt.imshow(overlay, alpha=0.5)
            plt.title(f"Low Opacity Regions (Lowest 50%) - Frame {frame_id} Cam {cam_id}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"overlay_frame_{frame_id}_cam_{cam_id}.png"))
            plt.close()


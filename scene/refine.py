import os
import random
import logging
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scene import GaussianModel, EnvLight
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import visualize_depth
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from typing import List, Tuple, Iterator

class ShuffledCyclicIterator:
    """
    循环打乱迭代器：每次遍历完所有元素后，重新打乱顺序再次遍历
    """
    
    def __init__(self, frame_ids: List[int], cam_ids: List[int], max_iterations: int, seed: int = None):
        """
        Args:
            frame_ids: 帧ID列表
            cam_ids: 相机ID列表  
            max_iterations: 最大迭代次数
            seed: 随机种子（可选）
        """
        self.frame_ids = frame_ids
        self.cam_ids = cam_ids
        self.max_iterations = max_iterations
        self.current_iteration = 1 # 当前迭代次数，从1开始
        self.cycle_count = 0
        
        # 创建所有(frame_id, cam_id)组合
        self.all_combinations = [(f, c) for f in frame_ids for c in cam_ids]
        self.total_combinations = len(self.all_combinations)
        
        # 当前周期的序列
        self.current_sequence = []
        self.sequence_index = 0
        
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
        
        # 初始化第一个序列
        self._shuffle_sequence()
        
        logging.info(f"ShuffledCyclicIterator initialized:")
        logging.info(f"  Total combinations: {self.total_combinations}")
        logging.info(f"  Max iterations: {max_iterations}")
        logging.info(f"  Estimated cycles: {(max_iterations + self.total_combinations - 1) // self.total_combinations}")
    
    def _shuffle_sequence(self):
        """重新打乱序列"""
        self.current_sequence = self.all_combinations.copy()
        random.shuffle(self.current_sequence)
        self.sequence_index = 0
        self.cycle_count += 1
        
        # logging.info(f"Cycle {self.cycle_count}: Shuffled sequence with {len(self.current_sequence)} combinations")
        # logging.info(f"  First 5 combinations: {self.current_sequence[:5]}")
    
    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        """返回迭代器自身"""
        return self
    
    def __next__(self) -> Tuple[int, int, int]:
        """
        返回下一个(frame_id, cam_id, iteration)元组
        
        Returns:
            Tuple[int, int, int]: (frame_id, cam_id, current_iteration)
        """
        if self.current_iteration >= self.max_iterations:
            raise StopIteration
        
        # 如果当前序列遍历完了，重新打乱
        if self.sequence_index >= len(self.current_sequence):
            self._shuffle_sequence()
        
        # 获取当前组合
        frame_id, cam_id = self.current_sequence[self.sequence_index]
        
        # 更新索引
        self.sequence_index += 1
        self.current_iteration += 1
        
        return frame_id, cam_id, self.current_iteration
    
    def get_progress_info(self) -> dict:
        """获取当前进度信息"""
        cycle_progress = self.sequence_index / len(self.current_sequence) if len(self.current_sequence) > 0 else 0
        total_progress = self.current_iteration / self.max_iterations
        
        return {
            'current_iteration': self.current_iteration,
            'max_iterations': self.max_iterations,
            'current_cycle': self.cycle_count,
            'cycle_progress': cycle_progress,
            'total_progress': total_progress,
            'combinations_per_cycle': len(self.current_sequence),
            'current_sequence_index': self.sequence_index
        }
    
    def skip_to_iteration(self, target_iteration: int):
        """跳转到指定迭代次数（用于恢复训练）"""
        if target_iteration <= 0 or target_iteration > self.max_iterations:
            return
        
        # 计算需要跳过多少个完整周期
        complete_cycles = (target_iteration - 1) // self.total_combinations
        remaining_steps = (target_iteration - 1) % self.total_combinations
        
        # 重新生成指定数量的周期
        for _ in range(complete_cycles + 1):
            self._shuffle_sequence()
        
        # 设置当前位置
        self.sequence_index = remaining_steps + 1
        self.current_iteration = target_iteration
        
        logging.info(f"Skipped to iteration {target_iteration} (cycle {self.cycle_count}, index {self.sequence_index})")


def refine_gaussians(args, dataset, gaussians, render_func, render_wrapper, background, tb_writer=None):
    """Refine static Gaussians with training iterations."""
    logging.info("Starting refinement process...")
    

    if args.refine_env_map_res > 0:
        env_map = EnvLight(resolution=args.refine_env_map_res).cuda()
        env_map.training_setup(args)
        logging.info(f"Environment map enabled with resolution {args.env_map_res}")
    else:
        env_map = None
        logging.info("Environment map disabled")
    
    # Setup optimizer (reinitialize learning rate).
    gaussians.training_setup(args)
    if env_map is not None:
        env_map.training_setup(args)
    
    # Create shuffled cyclic iterator.
    frame_ids = list(range(dataset.frame_num))
    cam_ids = list(range(dataset.cam_num))
    iterator = ShuffledCyclicIterator(
        frame_ids=frame_ids,
        cam_ids=cam_ids,
        max_iterations=args.refine_iter,
        seed=getattr(args, 'seed', None)
    )
    logging.info("Using standard shuffled iterator")
   
    
    # Create refine output directory.
    refine_path = os.path.join(args.model_path, 'refine')
    os.makedirs(refine_path, exist_ok=True)
    
    # Precompute operation schedule.
    def calculate_operation_schedule(args, refine_iter):
        """Precompute densify, prune, and opacity reset steps."""
        schedule = {
            'densify': [],
            'prune': [],
            'opacity_reset': []
        }
        
        # Densify schedule.
        for i in range(args.refine_densify_from_iter+1, args.refine_densify_until_iter):
            if i % args.refine_densification_interval == 0:
                schedule['densify'].append(i)
        
        # Prune schedule.
        for i in range(args.refine_prune_from_iter+1, args.refine_prune_until_iter):
            if i % args.refine_prune_interval == 0:
                schedule['prune'].append(i)
        
        # Opacity reset schedule.
        for i in range(args.refine_prune_from_iter, args.refine_prune_until_iter):
            if i % args.refine_opacity_reset_interval == 0:  
                schedule['opacity_reset'].append(i)
        
        return schedule
    
    operation_schedule = calculate_operation_schedule(args, args.refine_iter)
    logging.info(f"Operation schedule calculated:")
    logging.info(f"  Densify iterations: {operation_schedule['densify']}")
    logging.info(f"  Prune iterations: {operation_schedule['prune']}")
    logging.info(f"  Opacity reset iterations: {operation_schedule['opacity_reset']}")
    
    
    # Start refinement loop.
    ema_dict_for_log = defaultdict(float)
    progress_bar = tqdm(range(1, args.refine_iter + 1), desc="Refining", bar_format='{l_bar}{bar:50}{r_bar}')
    
    for frame_id, cam_id, iteration in iterator:
        # Update learning rate.
        gaussians.update_learning_rate(iteration)

        # Get viewpoint camera.
        viewpoint_cam = dataset.get_cam(dataset.resolution_scales[0], frame_id, cam_id)
        
        # Render and compute loss.
        loss, log_dict, render_pkg = render_wrapper(
            args, viewpoint_cam, gaussians, background, env_map, init=True, iteration=iteration
        )
        
        loss.backward()
        
        # Increase SH degree.
        if args.sh_increase_interval>0 and (iteration % args.sh_increase_interval == 0) and iteration > 0:
            gaussians.oneupSHdegree()
        
        if (iteration <= args.refine_densify_until_iter) and (gaussians.get_xyz.shape[0] < args.refine_max_points) :
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            
            # Update max radius for pruning.
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], 
                radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            # Densify (conservative settings).
            if iteration in operation_schedule['densify']:
                before_densify_count = gaussians.get_xyz.shape[0]
                gaussians.densify(args.refine_densify_grad_threshold, dataset.cameras_extent, init=True)
                after_densify_count = gaussians.get_xyz.shape[0]
                logging.info(f"Iteration {iteration}: Densified {after_densify_count - before_densify_count} points")

            # Prune.
            if iteration in operation_schedule['prune']:
                size_threshold = args.refine_size_threshold if (iteration > args.refine_opacity_reset_interval and args.refine_prune_big_point > 0) else None
                before_prune_count = gaussians.get_xyz.shape[0]
                gaussians.prune(dataset.cameras_extent, size_threshold, args.refine_thresh_opa_prune)
                after_prune_count = gaussians.get_xyz.shape[0]
                logging.info(f"Iteration {iteration}: Pruned {before_prune_count - after_prune_count} points")

            # Reset opacity.
            if iteration in operation_schedule['opacity_reset']:
                gaussians.reset_opacity()
                logging.info(f"Iteration {iteration}: Reset opacity for all points")



        # Optimization step.
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # Environment map optimization.
        if env_map is not None:
            env_map.optimizer.step()
            env_map.optimizer.zero_grad(set_to_none=True)
        

        
        # Logging and visualization.
        with torch.no_grad():
            torch.cuda.synchronize()
            log_dict['total_points'] = gaussians.get_xyz.shape[0]
            
            # EMA smoothing.
            for key in ["psnr", "loss"]:
                if key in log_dict:
                    ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]
            
            # Update progress bar.
            if iteration % 10 == 0:
                postfix = {
                    k[5:] if k.startswith("loss_") else k: f"{ema_dict_for_log[k]:.3f}" 
                    for k in ["psnr", "loss"] if k in ema_dict_for_log
                }
                postfix["pts"] = gaussians.get_xyz.shape[0]
                postfix["frame"] = f"{frame_id}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            
            # Tensorboard logs.
            if tb_writer and iteration % 50 == 0:
                for key, value in log_dict.items():
                    if isinstance(value, (float, int)):
                        tb_writer.add_scalar(f"refine/{key}", value, iteration)
                tb_writer.flush()
            
            # Periodic visualization.
            if iteration % args.refine_vis_interval == 0 or iteration == 1 or iteration == args.refine_iter:
                save_refine_visualization(render_pkg, viewpoint_cam, refine_path, iteration, frame_id, cam_id)

            if iteration % args.refine_save == 0 or iteration == args.refine_iter:
                # Save current model state.
                refine_ckpt_path = os.path.join(refine_path, f"init_gaussians_{iteration:05d}.pth")
                torch.save((gaussians.capture(), iteration), refine_ckpt_path)
                logging.info(f"Saved checkpoint at: {refine_ckpt_path}")
                
                if env_map is not None:
                    env_ckpt_path = os.path.join(refine_path, f"init_env_map_{iteration:05d}.pth")
                    torch.save((env_map.capture(), iteration), env_ckpt_path)
                    logging.info(f"Saved environment map at: {env_ckpt_path}")
                
                vis_dir = os.path.join(refine_path, f"{iteration}_vis_all")
                visualize_all_frames(args, gaussians, dataset, env_map, background, render_func, vis_dir, "final")
        
        # Clear cache.
        torch.cuda.empty_cache()
    
    # Refinement completed.
    logging.info(f"Over-filtered GS training completed!")
    refine_ckpt_path = os.path.join(args.model_path, "init_gaussians.pth")
    torch.save((gaussians.capture(), args.refine_iter), refine_ckpt_path)
    logging.info(f"Initial model saved to: {refine_ckpt_path}")
    
    if env_map is not None:
        env_ckpt_path = os.path.join(args.model_path, "init_env_map.pth")
        torch.save((env_map.capture(), args.refine_iter), env_ckpt_path)
        logging.info(f"Initial environment map saved to: {env_ckpt_path}")
    
    return gaussians, env_map


def visualize_all_frames(args, gaussians, dataset, env_map, background, render_func, save_dir, prefix=""):
        """Render and save visualizations for all frames."""
        os.makedirs(save_dir, exist_ok=True)
        
        for frame_id in range(dataset.frame_num):
            for cam_id in range(dataset.cam_num):
                # Get viewpoint camera.
                viewpoint_cam = dataset.get_cam(dataset.resolution_scales[0], frame_id, cam_id)
                
                with torch.no_grad():
                    render_pkg = render_func(viewpoint_cam, gaussians, args, background, 
                                        env_map=env_map, is_training=False)
                    
                    save_comparison_visualization(render_pkg, viewpoint_cam, save_dir, 
                                                frame_id, cam_id, prefix)

def save_comparison_visualization(render_pkg, viewpoint_cam, save_dir, frame_id, cam_id, prefix=""):
    """
    Compare the over-filtered static GS field with the original image for a specific frame and camera, and save the visualization.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        alpha = torch.clamp(render_pkg["alpha"], 0.0, 1.0)
        depth = render_pkg["depth"]
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        normal = render_pkg['normal']

        sky_mask_bool = (viewpoint_cam.sky_mask.cuda()>0) if viewpoint_cam.sky_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)
        pts_depth = viewpoint_cam.pts_depth.cuda() if viewpoint_cam.pts_depth is not None else torch.zeros_like(depth)
        pts_depth[sky_mask_bool] = 0.0  
        depth[sky_mask_bool] = 0.0 
        
        depth_vis = visualize_depth(depth)
        pts_depth_vis = visualize_depth(pts_depth)
        
        grid = make_grid([
            gt_image,
            image,
            depth_vis,
            pts_depth_vis,
            normal,
            alpha.repeat(3, 1, 1),
        ], nrow=6)
        
        filename = f"{prefix}_f{frame_id:03d}_c{cam_id}.png"
        save_image(grid, os.path.join(save_dir, filename))
        
    except Exception as e:
        logging.warning(f"Failed to save comparison visualization for frame {frame_id}: {e}")


def save_refine_visualization(render_pkg, viewpoint_cam, refine_path, iteration, frame_id, cam_id):
    """
    Save visualization during refinement for a specific frame and camera, including the rendered image, depth map, normal map, and alpha mask, compared to the original image.
    """
    try:
        alpha = torch.clamp(render_pkg["alpha"], 0.0, 1.0) 
        depth = render_pkg["depth"]
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        normal = render_pkg['normal']


        sky_mask_bool = (viewpoint_cam.sky_mask.cuda()>0) if viewpoint_cam.sky_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)
        depth[sky_mask_bool] = 0.0  
        
        depth_vis = visualize_depth(depth)
        semantic_mask = viewpoint_cam.semantic_mask.cuda().repeat(3, 1, 1) if viewpoint_cam.semantic_mask is not None else torch.zeros_like(alpha.repeat(3, 1, 1)) 
       
        grid = make_grid([
            gt_image,
            image,
            depth_vis,
            semantic_mask,
            normal,
            alpha.repeat(3, 1, 1),
        ], nrow=6)
        
        filename = f"refine_{iteration:05d}_f{frame_id}_c{cam_id}.png"
        save_image(grid, os.path.join(refine_path, filename))
        
    except Exception as e:
        logging.warning(f"Failed to save visualization at iteration {iteration}: {e}")

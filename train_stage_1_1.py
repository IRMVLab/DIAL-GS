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
from collections import defaultdict
import torch
import torch.nn.functional as F
import random
import numpy as np
from gaussian_renderer import get_renderer
from scene import Scene, GaussianModel, EnvLight
from scene.refine import refine_gaussians
from utils.general_utils import seed_everything, visualize_depth, init_logging
from utils.mapping_utils import back_project_from_depth,collect_pcd,compute_density_map
from utils.graphics_utils import BasicPointCloud
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
import logging
from omegaconf import OmegaConf
from scene.waymo_static_loader import WaymoDataset
from scene.kittimot_static_loader import KittiDataset
import cv2

dataset_dict = {
    "Waymo": WaymoDataset,
    "KittiMot": KittiDataset,
}


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False


EPS = 1e-5
non_zero_mean = (
    lambda x: sum(x) / len(x) if len(x) > 0 else -1
)


def suppress_semantic_overlaps(static_gaussians, viewpoint):
    """Project static Gaussians to the image plane and zero the ones inside semantic masks."""
    semantic_mask = getattr(viewpoint, "semantic_mask", None)
    if semantic_mask is None:
        return

    xyz = static_gaussians.get_xyz
    w2c = torch.inverse(viewpoint.c2w)
    K = viewpoint.get_k().float()

    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=1)
    xyz_cam = (w2c @ xyz_homo.T).T[:, :3]
    valid_depth = xyz_cam[:, 2] > 0
    if not valid_depth.any():
        return

    xyz_proj = (K @ xyz_cam[valid_depth].T).T
    xy_pixel = xyz_proj[:, :2] / xyz_proj[:, 2:3]
    x_pixel = torch.round(xy_pixel[:, 0]).long()
    y_pixel = torch.round(xy_pixel[:, 1]).long()

    if len(semantic_mask.shape) == 3:
        H, W = semantic_mask.shape[1], semantic_mask.shape[2]
        semantic_mask_2d = semantic_mask[0]
    else:
        H, W = semantic_mask.shape
        semantic_mask_2d = semantic_mask

    valid_pixel = (x_pixel >= 0) & (x_pixel < W) & (y_pixel >= 0) & (y_pixel < H)
    if not valid_pixel.any():
        return

    valid_x = torch.clamp(x_pixel[valid_pixel], 0, W - 1)
    valid_y = torch.clamp(y_pixel[valid_pixel], 0, H - 1)
    semantic_values = semantic_mask_2d[valid_y, valid_x]

    mask_indices = torch.nonzero(valid_depth, as_tuple=True)[0][valid_pixel]
    dynamic_mask_indices = mask_indices[semantic_values > 0]
    if len(dynamic_mask_indices) == 0:
        return

    with torch.no_grad():
        static_gaussians._opacity[dynamic_mask_indices] = torch.logit(
            torch.tensor(1e-6, device=static_gaussians._opacity.device)
        )

def train_warp_gs(frame_id, args, dataset, dynamic_gaussians, static_gaussians, env_map, render_func_merge, render_wrapper_merge, render_func, render_wrapper, background=None ,tb_writer=None):

    
    vis_path = os.path.join(args.model_path, 'visualization')
    os.makedirs(vis_path, exist_ok=True)
    
    # Preserve the original static Gaussian opacity so it can be restored later
    original_opacity = static_gaussians._opacity.clone().detach()
    
    static_pcd = BasicPointCloud(points= static_gaussians.get_xyz.detach().cpu().numpy(), # (N, 3
                                 colors= np.zeros_like(static_gaussians.get_xyz.detach().cpu().numpy())) # (N, 3)

    ema_dict_for_log = defaultdict(int)  # Used to show PSNR and related stats on the progress bar
    
    # 
    pcd_list_new = []
    for i in range(dataset.cam_num):
        viewpoint = dataset.get_cam(dataset.resolution_scales[0], frame_id, i)
        candidate_mask = viewpoint.semantic_mask.detach().cpu().numpy() # for back project depth points to 3D to create gs for dynamic objects
        gt_depth = viewpoint.gt_pts_depth.detach().cpu().numpy()
        valid_depth = (gt_depth > 0) & (gt_depth < args.valid_max_depth) 
        candidate_mask = candidate_mask & valid_depth 
        pcd = back_project_from_depth(args, viewpoint, candidate_mask, frame_id, i) # back project the candidate dynamic points to 3D 
        pcd_list_new.append(pcd)
        pcd_new = collect_pcd(pcd_list_new) 
        suppress_semantic_overlaps(static_gaussians, viewpoint) # Zero out static Gaussians that project inside semantic masks to avoid interference with dynamic object reconstruction


    # Merge static pcd and candidate pcd to create the initial Gaussian model for this frame
    dynamic_gaussians.create_from_pcd_merge(pcd_new, static_pcd, 1.0, args.init_opacity) # Create the dynamic point cloud model
    dynamic_gaussians.training_setup(args)

    scale_idx = 0 # Train at the original resolution without downsampling
    progress_bar = tqdm(range(1,args.iter_per_frame+1), bar_format='{l_bar}{bar:50}{r_bar}')
    
    for iteration in range(1, args.iter_per_frame + 1):   
        dynamic_gaussians.update_learning_rate(iteration)
        
        # Iterate over all cameras for this frame to render, compute loss, backprop, and densify
        log_dict_list = []
        render_pkg_list = [] # Store the render outputs for each camera
        
        for cam_id in range(dataset.cam_num):  
            viewpoint_cam = dataset.get_cam(dataset.resolution_scales[scale_idx],frame_id,cam_id) 
            loss, log_dict, render_pkg = render_wrapper(args, viewpoint_cam, dynamic_gaussians, background, env_map=None, init=False, iteration=iteration)
            loss.backward()
            
            # Store render outputs for later visualization
            if iteration % args.vis_interval == 0 or iteration == 1 or iteration == args.iter_per_frame:
                render_pkg_list.append(render_pkg)
            
            log_dict['loss'] = loss.item() # Ensure the loss value is recorded in log_dict
            log_dict_list.append(log_dict.copy())

            # Densification and pruning
            if iteration <= args.densify_until_iter and (args.densify_until_num_points < 0 or dynamic_gaussians.get_xyz.shape[0] < args.densify_until_num_points):
                static_num = static_gaussians.get_xyz.shape[0]
                viewspace_point_tensor = render_pkg["viewspace_points"]# 2D gradients for each Gaussian
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                # Keep track of max radii in image-space for pruning
                dynamic_gaussians.max_radii2D[visibility_filter] = torch.max(dynamic_gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                dynamic_gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) # Update the accumulated 2D gradients
                
                if iteration >= args.densify_from_iter and iteration % args.densification_interval == 0:
                    # Densify the point cloud
                    dynamic_gaussians.densify(args.densify_grad_threshold,dataset.cameras_extent)
                    
                    # Prune oversized Gaussians
                    size_threshold = args.size_threshold if (iteration > args.opacity_reset_interval and args.prune_big_point > 0) else None
                    if size_threshold is not None:
                        size_threshold = size_threshold // dataset.resolution_scales[scale_idx]
                    dynamic_gaussians.prune(dataset.cameras_extent, size_threshold ,args.thresh_opa_prune)

                if (args.opacity_reset_interval >0) and (iteration % args.opacity_reset_interval == 0) or (args.white_background and iteration == args.densify_from_iter):
                    # Reset opacity
                    dynamic_gaussians.reset_opacity()

            dynamic_gaussians.optimizer.step()
            dynamic_gaussians.optimizer.zero_grad(set_to_none = True)

            torch.cuda.empty_cache()
        

        # Visualization and logging
        with torch.no_grad():
            log_dict_avg = {key: 0 for key in log_dict_list[0].keys()}
            for log_dict in log_dict_list:
                for key in log_dict.keys():
                    log_dict_avg[key] += log_dict[key] / len(log_dict_list)
            # Use the average metrics from all cameras to represent this iteration
            log_dict = log_dict_avg
            for key in ["psnr"]: # 'loss', "loss_l1", 
                ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]
                # log_dict[key] is the current value; ema_dict_for_log[key] stores the previous EMA
            
            torch.cuda.synchronize() 
            log_dict['total_points'] = dynamic_gaussians.get_xyz.shape[0]
            
            if iteration % 2 == 0:
                postfix = {k[5:] if k.startswith("loss_") else k:f"{ema_dict_for_log[k]:.{2}f}" for k, v in ema_dict_for_log.items()}
                postfix["scale"] = dataset.resolution_scales[scale_idx]
                postfix["pts"] = dynamic_gaussians.get_xyz.shape[0]
                progress_bar.set_postfix(postfix)
                progress_bar.update(2)
            
            # Record metrics to TensorBoard, using a dedicated folder per frame
            if tb_writer:
                for key, value in log_dict.items():
                    assert isinstance(value, (float, int)), f"{key} is not a number: {value}"
                    tb_writer.add_scalar(f"frame_{frame_id}/{key}", value, iteration)
                    tb_writer.flush()
            
            # Visualize Gaussian density maps
            if iteration % args.vis_interval == 0 or iteration == 1 or iteration == args.iter_per_frame:
                cam_list = [dataset.get_cam(dataset.resolution_scales[scale_idx], frame_id, i) for i in range(dataset.cam_num)]
                density_maps = []
                for cam in cam_list:
                    c2w = cam.c2w
                    w2c = torch.inverse(c2w) # World-to-camera transform
                    K = cam.get_k().cpu().numpy() # Camera intrinsics for this view
                    image_size = (cam.image_height, cam.image_width) # Current camera resolution
                    density_map = compute_density_map(dynamic_gaussians.get_xyz, w2c, K, image_size, point_radius=1.0, normalize=True)
                    density_maps.append(density_map)
                save_visualization_all(render_pkg_list, cam_list, vis_path, frame_id, density_maps, iteration, env_map=env_map) 

    # Warp instant GS to later frames for further consistency check
    warp_vis_path = os.path.join(args.model_path, f"detection", f"warp_{frame_id}")
    os.makedirs(warp_vis_path, exist_ok=True)
    for idx in range(frame_id+1, frame_id + args.warp_frame+1):
        with torch.no_grad():
            for cam_id in range(dataset.cam_num):
                viewpoint_cam = dataset.get_cam(1,idx,cam_id) 
                render_pkg = render_func_merge(viewpoint_cam, static_gaussians, dynamic_gaussians, args, background, env_map=env_map, is_training=False)
                warp_rgb = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0) # (H, W, 3)
                gt_rgb = viewpoint_cam.original_image.cpu().numpy().transpose(1, 2, 0) # (H, W, 3)
                # Save the warped RGB and ground-truth RGB frames
                warp_rgb = np.clip(warp_rgb, 0, 1)
                gt_rgb = np.clip(gt_rgb, 0, 1)
                warp_rgb = cv2.cvtColor(warp_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
                gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(warp_vis_path, f"warp_{idx}_cam{cam_id}.png"), (warp_rgb * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(warp_vis_path, f"gt_{idx}_cam{cam_id}.png"), (gt_rgb * 255).astype(np.uint8))

    # Restore the original static Gaussian opacity values
    static_gaussians._opacity.data = original_opacity


def save_visualization_all(render_pkg_list, cam_list, vis_path, frame_id, density_maps, iteration,env_map=None):
    """
    Save visualization grids where each row corresponds to a rendered variant and each column to a camera view.
    Top-to-bottom rows: ground-truth RGB, rendered RGB, pts_depth (GT depth), depth (rendered depth), alpha, v_map, not_sky_mask.
    """
    rows = []  # Store the images for each row
    for cam_id in range(len(cam_list)):  # Iterate through every camera
        render_pkg = render_pkg_list[cam_id]
        viewpoint_cam = cam_list[cam_id]

        # Extract rendered outputs
        alpha = torch.clamp(render_pkg["alpha"], 0.0, 1.0)
        depth = render_pkg["depth"]
        image = render_pkg["render"]
        not_sky_mask = torch.logical_not(viewpoint_cam.sky_mask[:1]).float() if viewpoint_cam.sky_mask is not None else torch.zeros_like(alpha)
        normal = render_pkg['normal']
        
        # Extract ground-truth RGB
        gt_image = viewpoint_cam.original_image.cuda()
        sky_mask_bool = (viewpoint_cam.sky_mask.cuda()>0) if viewpoint_cam.sky_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)
        # pts_depth = viewpoint_cam.pts_depth.cuda() if viewpoint_cam.pts_depth is not None else torch.zeros_like(depth) # dense depth
        pts_depth = viewpoint_cam.gt_pts_depth.cuda() if viewpoint_cam.gt_pts_depth is not None else torch.zeros_like(depth) # sparse depth
        pts_depth[sky_mask_bool] = 0.0  # Zero out sky regions
        depth[sky_mask_bool] = 0.0  # Zero out sky regions in the rendered depth
        
        # Visualize depth maps
        depth_vis = visualize_depth(depth)
        pts_depth_vis = visualize_depth(pts_depth)
        gt_rgb_depth_overlay = 0.4 * gt_image + 0.6 * pts_depth_vis  # Simple weighted fusion
        gt_rgb_depth_overlay = torch.clamp(gt_rgb_depth_overlay, 0.0, 1.0)
        
        # Visualize the density map for debugging
        if density_maps[cam_id] is not None:
            density_map_vis = visualize_depth(density_maps[cam_id].unsqueeze(0), near=0.01, far=1.0)
        else:
            density_map_vis = torch.zeros_like(depth_vis)
       

        # Build each column for the grid
        col = [
            gt_image,  # Ground-truth RGB
            image,  # Rendered RGB
            pts_depth_vis,  # Ground-truth depth
            gt_rgb_depth_overlay,  # GT depth blended with GT RGB
            depth_vis,  # Rendered depth
            normal,  # Rendered normals
            density_map_vis, # Gaussian density visualization
            alpha.repeat(3, 1, 1),  # Opacity
            not_sky_mask.repeat(3, 1, 1),  # Non-sky mask
        ]
        rows.append(col)

    # Swap the first two rows for Waymo if needed
    # rows[0], rows[1] = rows[1], rows[0]  # Swap row order when required

    # Stitch each column into rows
    grid = make_grid([img for row in zip(*rows) for img in row], nrow=len(cam_list))

    # Save the visualization grid
    downsampled_grid = F.interpolate(grid.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
    save_image(downsampled_grid, os.path.join(vis_path, f"frame_{frame_id}_{iteration:03d}.png"))  


def save_visualizations(render_pkg, viewpoint_cam, vis_path, iteration):
    alpha = torch.clamp(render_pkg["alpha"], 0.0, 1.0)
    depth = render_pkg["depth"]
    image = render_pkg["render"]
    rendered_normal = torch.clamp(render_pkg.get("normal", torch.zeros_like(image)), 0.0, 1.0)
    sky_mask = viewpoint_cam.sky_mask.cuda() if viewpoint_cam.sky_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)
    gt_image = viewpoint_cam.original_image.cuda()
    gt_normal = viewpoint_cam.normal_map.cuda() if viewpoint_cam.normal_map is not None else torch.zeros_like(rendered_normal)
    pseudo_normal = torch.clamp(render_pkg.get("depth_normal", gt_normal), 0.0, 1.0)
    # other_img = []
    feature = render_pkg['feature'] / alpha.clamp_min(1e-5) # TODO: clarify the rationale for dividing by alpha
    t_map = feature[0:1] # (1, H, W) # Beta map controlling transparency decay: larger beta decays more slowly
    v_map = feature[1:] # (1, H, W)
    v_norm_map = v_map.norm(dim=0, keepdim=True)
    et_color = visualize_depth(t_map, near=0.01, far=1)
    v_color = visualize_depth(v_norm_map, near=0.01, far=1)
    # other_img.append(et_color)
    # other_img.append(v_color)
    # dynamic_mask = render_pkg['dynamic_mask'] 
    dynamic_mask = torch.zeros_like(alpha, dtype=torch.bool) 

    
    if viewpoint_cam.pts_depth is not None:
        pts_depth = viewpoint_cam.pts_depth # Depth from projected point clouds
        # pts_depth[pts_depth == 0] = 900
        pts_depth_vis = visualize_depth(pts_depth)
        # other_img.append(pts_depth_vis)
    
    not_sky_mask = torch.logical_not(sky_mask[:1]).float()  
    rendered_distance = render_pkg.get("rendered_distance", torch.zeros_like(alpha))
    # d_mask = render_pkg.get("d_mask", torch.zeros_like(alpha))
    grid = make_grid([
        image, # Rendered RGB
        alpha.repeat(3, 1, 1), # Rendered opacity
        visualize_depth(depth), # Rendered depth map
        rendered_normal,# Rendered normal map
        et_color, # Timing/beta visualization
        visualize_depth(rendered_distance), 
        # dino_cosine.repeat(3, 1, 1), 
        gt_image,
        not_sky_mask.repeat(3, 1, 1),
        pts_depth_vis,
        # gt_normal * not_sky_mask,
        pseudo_normal,
        dynamic_mask.repeat(3, 1, 1), # Dynamic mask (only available in stage two)
        v_color, # Velocity magnitude visualization
        # d_mask.repeat(3, 1, 1),
    ], nrow=6)

    # [ image | alpha | depth | rendered_normal | t_map | dist_vis ]
    # [ gt_img | sky_mask  | pts_depth | pseudo_normal | dyn_mask | v_map  ]

    save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default = "configs/base.yaml")
    cli_args, _ = parser.parse_known_args()

    # Load configuration files
    base_conf = OmegaConf.load(cli_args.base_config)
    second_conf = OmegaConf.load(cli_args.config)
    cli_conf = OmegaConf.from_cli()


    # Merge configurations
    args = OmegaConf.merge(base_conf, second_conf, cli_conf) # Later configs overwrite earlier ones
    
    os.makedirs(args.model_path, exist_ok=True)    
    init_logging(os.path.join(args.model_path, "train.log"))
    OmegaConf.save(args, os.path.join(args.model_path, "config.yaml"))
    seed_everything(args.seed)

    if args.scene_type == "KittiMot" :
        
        if args.source_path.split("/")[-1]=="0001":
            args.start_frame = 380 
            args.end_frame = 431  #431 the last frame is included
        elif args.source_path.split("/")[-1]=="0002":
            args.start_frame = 140 #140
            args.end_frame = 224 # 232
        elif args.source_path.split("/")[-1]=="0006":
            args.start_frame = 65
            args.end_frame = 120 #126
        print("#"*20)
        print(args.source_path.split("/")[-1])
        print("start frame: ", args.start_frame)
        print("end frame: ", args.end_frame)
        print("#"*20)

    dataset = dataset_dict[args.scene_type](args)  # Load the dataset class specified in the config
    dataset.load()

    
    # Initialize renderers
    render_func, render_wrapper, render_func_merge, render_wrapper_merge = get_renderer(args.render_type)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        logging.info("Logging progress to Tensorboard")
    else:
        tb_writer = None
        logging.info("Tensorboard not available: not logging progress")



    # Over-filtered Ststic Scene: use semantic masks to suppress potential dynamic objects during reconstruction
    if args.resume  < 0:
        logging.info("Starting first stage: semantic mask and reconstruction")
        points_all = dataset.pcd
        color_all = np.zeros_like(points_all)  # (N, 3) with no color information
        lidar_pcd = BasicPointCloud(
            points = points_all,
            colors = color_all,
            normals = None,  # No normals provided
            time = None  # No timestamps provided
        )

        pcd_gaussians = GaussianModel(args)
        pcd_gaussians.create_from_pcd(lidar_pcd, 1, args.refine_init_opacity)
        static_gaussians,static_env_map = refine_gaussians(args, dataset, pcd_gaussians, render_func, render_wrapper, background, tb_writer)


    else:
        static_gaussians = GaussianModel(args)
        static_env_map = EnvLight(resolution=args.refine_env_map_res).cuda()
        checkpoint = os.path.join(args.model_path, "init_gaussians.pth")
        model_params, frame_id_start = torch.load(checkpoint) 
        static_gaussians.restore(model_params, args)
        static_gaussians.save_ply(os.path.join(args.model_path, f"static_gaussians.ply"))
        env_checkpoint = os.path.join(args.model_path, "init_env_map.pth")
        (light_params, _) = torch.load(env_checkpoint)
        static_env_map.restore(light_params)


    # - Train the dynamic candidate Gaussians. 
    # - Combine over-filtered static Gaussians with candidate dynamic Gaussians to instant GS filed
    # - Warp the instant GS to later frames for further consistency check and dynamic object detection
    for i in range(0, dataset.frame_num-args.warp_frame-1):
        logging.info(f"Inconsistance Check Frame {i}...")
        dynamic_gaussians = GaussianModel(args)
        train_warp_gs(i, args, dataset, dynamic_gaussians, static_gaussians, static_env_map, render_func_merge, render_wrapper_merge, render_func, render_wrapper,background, tb_writer)
        del dynamic_gaussians
        torch.cuda.empty_cache()
    

    # All done
    logging.info("Warpping complete.")

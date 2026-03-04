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
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.dynamic_gaussian_model import DynamicGaussianModel
from scene.dynamic_model import scale_grads
from scene.cameras import Camera
from utils.sh_utils import eval_sh
import torch.nn.functional as F
import numpy as np
import kornia
from utils.loss_utils import psnr, ssim, tv_loss, lncc, loss_cls_3d_dynamic_static,loss_velocity_consistency,loss_velocity_consistency_fast
from utils.graphics_utils import patch_offsets, patch_warp, render_normal
from utils.general_utils import compute_camera_frustum_corners, compute_frustum_point_ids,get_mask_from_projection
import os
import cv2
from torchvision.utils import save_image, make_grid
import time
EPS = 1e-5


def render_pvg(
    viewpoint_camera: Camera,
    pc: DynamicGaussianModel,
    pipe,  # args
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    env_map=None,
    time_shift=None,
    other=[],
    mask=None,
    is_training=False,
    return_depth_normal=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if env_map is not None else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None

    if time_shift is not None:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp - time_shift)
        means3D = means3D + pc.get_inst_velocity * time_shift
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp - time_shift)
    else:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
    opacity = opacity * marginal_t

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, pc.get_max_sh_channels
            )
            dir_pp = (
                means3D.detach()
                - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            ).detach()
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    feature_list = other # [t_scale, v] (N,1) (N,3)
   
    # TODO: MOVE TO MASKED MEANS3D
    # normals = pc.get_normal(viewpoint_camera.c2w, means3D, from_scaling=False)
    normals = pc.get_normal_v2(viewpoint_camera, means3D) # (N,3) normal of the flattened axis for the 2D Gaussian
    # Transform normals to camera space
    normals = normals @ viewpoint_camera.world_view_transform[:3, :3]

    if len(feature_list) > 0:
        features = torch.cat(feature_list, dim=1) # (N,S_other)
        S_other = features.shape[1]
    else:
        features = torch.zeros_like(means3D[:, :0]) # (N,)
        S_other = 0
   
    features = torch.cat([features, pc.get_obj_dc.squeeze(1)],dim=1)
    S_other += pc.num_objects


    # Prefilter
    if mask is None:
        mask = marginal_t[:, 0] > 0.05
    else:
        mask = mask & (marginal_t[:, 0] > 0.05)

    pts_in_cam = (
        means3D @ viewpoint_camera.world_view_transform[:3, :3]
        + viewpoint_camera.world_view_transform[3, :3]
    )
    depth_z = pts_in_cam[:, 2:3]

    # Dot product between the point-to-camera vector and the normal measures how aligned the 2D Gaussian is to the view
    # When the normal faces the camera, local_distance equals the point-to-camera distance
    # When the normal is perpendicular to the camera, local_distance becomes 0
    local_distance = (normals * pts_in_cam).sum(-1).abs() # (N,3)*(N,3) -> (N)
    
    depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device) # (N,2)
    depth_alpha[mask] = torch.cat([depth_z, torch.ones_like(depth_z)], dim=1)[mask] # depth_alpha is (z,1) for masked points, otherwise (0,0)
    
    # print("Shape of features:", features.shape) # (N, S_other)  
    # print("Shape of depth_alpha:", depth_alpha.shape) # (N, 2)
    # print("Shape of normals:", normals.shape) # (N, 3)
    # print("Shape of local_distance:", local_distance.shape) # (N,)

    # features = torch.cat([features, depth_alpha, normals, local_distance.unsqueeze(-1), pc.get_obj_dc.squeeze(1)], dim=1) # (N, S_other + 1 + 1 + 3 + 1)
    features = torch.cat([features, depth_alpha, normals, local_distance.unsqueeze(-1)],dim=1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    contrib, rendered_image, rendered_feature, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        features=features,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        mask=mask,
    )

    rendered_other, rendered_depth, rendered_opacity, rendered_normal, rendered_distance = (
        rendered_feature.split([S_other, 1, 1, 3, 1], dim=0)
    )

    rendered_normal = rendered_normal * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    rendered_image_before = rendered_image
    if env_map is not None:
        bg_color_from_envmap = env_map(
            viewpoint_camera.get_world_directions(is_training).permute(1, 2, 0)
        ).permute(2, 0, 1)
        rendered_image = rendered_image + (1 - rendered_opacity) * bg_color_from_envmap

    return_dict = {
        "render": rendered_image,
        "render_nobg": rendered_image_before,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "contrib": contrib,
        "depth": rendered_depth,
        "alpha": rendered_opacity,
        "normal": rendered_normal,
        "feature": rendered_other,
        "scaling": scales,
        "rendered_distance": rendered_distance,
    }
   
   # Depth-normal visualization
    if return_depth_normal:
        depth_normal = (
            render_normal(viewpoint_camera, rendered_depth.squeeze())
            * (rendered_opacity).detach()
        )
        return_dict.update({"depth_normal": depth_normal * 0.5 + 0.5})
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict


def calculate_loss(
    gaussians: DynamicGaussianModel,
    viewpoint_camera: Camera,
    bg_color: torch.Tensor,
    args,
    render_pkg: dict,
    env_map,
    iteration,
    camera_id,
    nearest_cam: Camera = None,
    dynamic_dict: dict = None,
):
    log_dict = {}

    image = render_pkg["render"]
    depth = render_pkg["depth"]
    alpha = render_pkg["alpha"] # (1, H, W)
    visibility_filter = render_pkg["visibility_filter"]
    feature = render_pkg["feature"] / alpha.clamp_min(EPS)
    t_map = feature[0:1]
    v_map = feature[1:4]
    object_dc = feature[4:] # (8, H, W)


    sky_mask = (
        viewpoint_camera.sky_mask.cuda()
        if viewpoint_camera.sky_mask is not None
        else torch.zeros_like(alpha, dtype=torch.bool)
    )

    dynamic_mask = (
        viewpoint_camera.dynamic_mask.cuda()
        if viewpoint_camera.dynamic_mask is not None
        else torch.zeros_like(alpha, dtype=torch.bool)
    )  # true for dynamic

    dynamic_mask = ~dynamic_mask  # true for static

    id_mask = (
        viewpoint_camera.id_mask.cuda()
        if viewpoint_camera.id_mask is not None
        else torch.full_like(alpha, -1, dtype=torch.int32)
    )  

    #(1, H, W)
    id_mask = torch.where(id_mask == -1, gaussians.num_classes-1, id_mask).long()  # last ID is the static class
    
    loss = 0.0

    sky_depth = 900
    depth = depth / alpha.clamp_min(EPS)
    if env_map is not None:
        if args.depth_blend_mode == 0:  # harmonic mean
            depth = 1 / (
                alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth
            ).clamp_min(EPS)
        elif args.depth_blend_mode == 1:
            depth = alpha * depth + (1 - alpha) * sky_depth

    # gt_image = viewpoint_camera.original_image.cuda()
    gt_image, gt_image_gray = viewpoint_camera.get_image()

    loss_l1 = F.l1_loss(image, gt_image, reduction="none")  # [3, H, W]
    loss_ssim = 1.0 - ssim(image, gt_image, size_average=False)  # [3, H, W]

    log_dict["loss_l1"] = loss_l1.mean().item()
    log_dict["loss_ssim"] = loss_ssim.mean().item()
    metrics = {}
    # loss_mult = torch.ones_like(depth, dtype=depth.dtype)
    # dino_part = torch.zeros_like(depth, dtype=depth.dtype)

    
    loss += (
        1.0 - args.lambda_dssim
    ) * loss_l1.mean() + args.lambda_dssim * loss_ssim.mean()

    psnr_for_log = psnr(image, gt_image).double()
    log_dict["psnr"] = psnr_for_log
    alpha_mask = alpha.data > EPS  # detached (1, H, W)
    
    # Depth supervision
    if args.lambda_lidar > 0:
        assert viewpoint_camera.gt_pts_depth is not None
        pts_depth = viewpoint_camera.gt_pts_depth.cuda()

        mask = pts_depth > 0
        loss_lidar = torch.abs(
            1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)
        ).mean()
        if args.lidar_decay > 0:
            iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
        else:
            iter_decay = 1
        log_dict["loss_lidar"] = loss_lidar.item()
        loss += iter_decay * args.lambda_lidar * loss_lidar


    # Normal supervision
    if args.lambda_normal > 0:
        rendered_normal = render_pkg["normal"]  # (3, H, W)
        if args.load_normal_map:
            gt_normal = viewpoint_camera.normal_map.cuda()
            loss_normal = F.l1_loss(rendered_normal * alpha_mask, gt_normal * alpha_mask)
            loss_normal += tv_loss(rendered_normal)
            log_dict["loss_normal"] = loss_normal.item()
            loss += args.lambda_normal * loss_normal
        elif "depth_normal" in render_pkg:
            depth_normal = render_pkg["depth_normal"]
            loss_normal = F.l1_loss(
                rendered_normal * alpha_mask, depth_normal * alpha_mask
            )
            loss_normal += tv_loss(rendered_normal)
            log_dict["loss_normal"] = loss_normal.item()
            loss += 0.1 * args.lambda_normal * loss_normal
    
    dynamic_mask_np = ~dynamic_mask.squeeze().cpu().numpy()  # (H, W) true for dynamic
    dilate_pixels = 5
    kernel = np.zeros((dilate_pixels+1, 1), dtype=np.uint8)
    kernel[:, 0] = 1
    dynamic_mask_np = cv2.dilate(dynamic_mask_np.astype(np.uint8), kernel, iterations=1,anchor=(0, 0)).astype(bool)
    dynamic_mask_for_pvg = torch.tensor(~dynamic_mask_np, dtype=torch.bool, device=alpha.device).unsqueeze(0)  # (1, H, W) true for static
    
    
    # Beta regularization
    if args.lambda_t_reg > 0 and args.enable_dynamic:
        if iteration > args.dynamic_mask_epoch:
            if args.scene_type == "Waymo":
                if viewpoint_camera.colmap_id % 5 ==0:
                    loss_t_reg = -torch.abs(t_map * dynamic_mask_for_pvg).sum()/dynamic_mask_for_pvg.sum() * 5
                else:
                    loss_t_reg = -torch.abs(t_map).mean()
            elif args.scene_type == "KittiMot":
                total_frame_num = args.end_frame - args.start_frame + 1
                if viewpoint_camera.colmap_id  < total_frame_num//2:
                    loss_t_reg = -torch.abs(t_map * dynamic_mask_for_pvg).sum()/dynamic_mask_for_pvg.sum() * 5
                else:
                    loss_t_reg = -torch.abs(t_map).mean()
            else:
                loss_t_reg = -torch.abs(t_map).mean()

            
            log_dict['loss_t_reg'] = loss_t_reg.item()
            loss += args.lambda_t_reg * loss_t_reg

    # Velocity regularization
    if args.lambda_v_reg > 0 and args.enable_dynamic:
        
        loss_v_reg = (torch.abs(v_map)).mean()
        # Add extra velocity_reg in major views
        if  iteration > args.dynamic_mask_epoch:
            if args.scene_type == "Waymo":
                if viewpoint_camera.colmap_id % 5 ==0:
                    loss_v_reg += (torch.abs(v_map) * dynamic_mask_for_pvg).sum()/dynamic_mask_for_pvg.sum() * 10 
            elif args.scene_type == "KittiMot":
                total_frame_num = args.end_frame - args.start_frame + 1
                if viewpoint_camera.colmap_id  < total_frame_num//2:
                    loss_v_reg += (torch.abs(v_map) * dynamic_mask_for_pvg).sum()/dynamic_mask_for_pvg.sum() * 10    
       
            
        log_dict["loss_v_reg"] = loss_v_reg.item()
        loss += args.lambda_v_reg * loss_v_reg

    # Inverse-depth smoothness loss (encourage depth gradients to follow RGB gradients)
    if args.lambda_inv_depth > 0:
        inverse_depth = 1 / (depth + 1e-5)
        loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
            inverse_depth[None], gt_image[None]
        )
        log_dict["loss_inv_depth"] = loss_inv_depth.item()
        loss = loss + args.lambda_inv_depth * loss_inv_depth

    # Velocity smoothness loss (encourage velocity gradients to follow RGB gradients)
    if args.lambda_v_smooth > 0 and args.enable_dynamic:
        loss_v_smooth = kornia.losses.inverse_depth_smoothness_loss(
            v_map[None], gt_image[None]
        )
        log_dict["loss_v_smooth"] = loss_v_smooth.item()
        loss = loss + args.lambda_v_smooth * loss_v_smooth

    # Beta smoothness loss (encourage beta gradients to follow RGB gradients)
    if args.lambda_t_smooth > 0 and args.enable_dynamic:
        loss_t_smooth = kornia.losses.inverse_depth_smoothness_loss(
            t_map[None], gt_image[None]
        )
        log_dict["loss_t_smooth"] = loss_t_smooth.item()
        loss = loss + args.lambda_t_smooth * loss_t_smooth
    
    # Sky opacity penalty (prefer lower opacity in sky regions)
    if args.lambda_sky_opa > 0:
        o = alpha.clamp(1e-6, 1 - 1e-6)
        sky = sky_mask.float()
        loss_sky_opa = (-sky * torch.log(1 - o)).mean()
        log_dict["loss_sky_opa"] = loss_sky_opa.item()
        loss = loss + args.lambda_sky_opa * loss_sky_opa

    # Opacity entropy loss (encourage a well-spread opacity distribution)
    if args.lambda_opacity_entropy > 0:
        o = alpha.clamp(1e-6, 1 - 1e-6)
        loss_opacity_entropy = -(o * torch.log(o)).mean()
        log_dict["loss_opacity_entropy"] = loss_opacity_entropy.item()
        loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy
    
    # Scale loss (encourage balanced scaling components)
    if args.lambda_scaling > 0:
        scaling = render_pkg["scaling"]  # (N, 3)
        scaling_loss = (
            (scaling - scaling.mean(dim=-1, keepdim=True)).abs().sum(-1).mean()
        )
        lambda_scaling = (
            args.lambda_scaling * 0.1
        )  # - 0.99 * args.lambda_scaling * min(1, 4 * iteration / args.iterations)
        log_dict["scaling_loss"] = scaling_loss.item()
        loss = loss + lambda_scaling * scaling_loss
    
    # 2D classification loss (enforce consistency with id_mask)
    if args.lambda_obj_2d > 0  and iteration> args.id_2d_epoch:
        start_2d_obj_loss = False
        if args.scene_type == "Waymo":
            if viewpoint_camera.colmap_id % 5 ==0:
                start_2d_obj_loss = True
        elif args.scene_type == "KittiMot":
            total_frame_num = args.end_frame - args.start_frame + 1
            if viewpoint_camera.colmap_id  < total_frame_num//2:
                start_2d_obj_loss = True
                
               
        if start_2d_obj_loss:
            invalid_mask = (id_mask < 0) | (id_mask >= gaussians.num_classes)
            if invalid_mask.any():
                raise ValueError(f"Invalid ID in id_mask for camera {viewpoint_camera.colmap_id}. Check if id_mask is correctly loaded and processed.")

            valid_mask = alpha_mask  # Only compute the loss where opacity is sufficient # (1, H, W)

            if valid_mask.sum() > 0:
                logits = gaussians.classifier(object_dc)  # (8, H, W) -> (21, H, W)
                ce_loss = gaussians.cls_criterion(
                    logits.unsqueeze(0),  # (1, 21, H, W)
                    id_mask,  # (1, H, W)
                )  # (1,H,W)
                ce_loss = (ce_loss * valid_mask.float()).squeeze().sum() / valid_mask.sum() # (1,H,W) -> (H,W)
                ce_loss = ce_loss / torch.log(torch.tensor(gaussians.num_classes))  # normalize to (0,1)
                loss_obj_2d = ce_loss
                log_dict["loss_obj_2d"] = loss_obj_2d.item()
                loss += args.lambda_obj_2d * loss_obj_2d


    # 3D classification loss
    if args.lambda_obj_3d > 0 and iteration>args.id_3d_epoch:
        start_3d_obj_loss = False
        if args.scene_type == "Waymo":
            if viewpoint_camera.colmap_id % 5 ==0:
                start_3d_obj_loss = True
                cam_no = 0
        elif args.scene_type == "KittiMot":
            total_frame_num = args.end_frame - args.start_frame + 1
            if viewpoint_camera.colmap_id  < total_frame_num//2:
                start_3d_obj_loss = True
                cam_no=2

        if start_3d_obj_loss and dynamic_dict is not None:

            opacity = gaussians.get_opacity # (N, 1)
            marginal_t = gaussians.get_marginal_t(viewpoint_camera.timestamp) # (N, 1)
            exist_mask = (marginal_t[:, 0].detach() > 0.05) & (opacity[:, 0].detach() > 0.01) # (N,) tensor
            
            # Focus on dynamic objects observed in the primary camera
            dynamic_ids = np.arange(len(dynamic_dict[f"cam_{cam_no}"])) # 0,1,2...
            dynamic_ids = torch.tensor(dynamic_ids, dtype=torch.int64, device=object_dc.device)  # 0,1,2...
            obj_dc_3d = gaussians.get_obj_dc.permute(2,0,1) # (N,1,8) -> (8,N,1)
            logits3d = gaussians.classifier(obj_dc_3d)  # (8, N, 1) -> (21, N, 1)
            


            with torch.no_grad():
                gaussian_points = gaussians.get_xyz.detach()  # (N, 3)
                prob_obj3d_detached = torch.softmax(logits3d,dim=0).detach().squeeze().permute(1,0)
                reliable_mask = (prob_obj3d_detached.max(dim=1).values > -1)
                id_pred_3d_detached = prob_obj3d_detached.argmax(dim=1)
                beta_mask = (gaussians.get_scaling_t[:,0].detach() < args.separate_scaling_t*1.5)  
                dynamic_mask_3d = torch.isin(id_pred_3d_detached.squeeze(),dynamic_ids) & exist_mask & reliable_mask & beta_mask # (N,) predicted dynamic, still alive, reliable, and with small beta
                static_mask_3d = get_mask_from_projection(gaussian_points, exist_mask, viewpoint_camera.get_k().float(), torch.inverse(viewpoint_camera.c2w), ~dynamic_mask_for_pvg.squeeze(), \
                                                     image_width=viewpoint_camera.image_width, image_height=viewpoint_camera.image_height) 
              
            prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0) # (N,21)
            velocity = gaussians.get_inst_velocity  # (N, 3)
            beta = gaussians.get_scaling_t  # (N, 1)
            loss_obj_3d, knn_mask = loss_cls_3d_dynamic_static(
                gaussian_points,
                prob_obj3d,
                dynamic_mask_3d,
                static_mask_3d,
                k=10,
                batch_size=20_000,
                detach_dynamic_prob=False
            )
        
        
            # # debug: visualize whether the existing/dynamic Gaussians are separated correctly
            # render_pkg_dynamic = render_pvg(viewpoint_camera, gaussians, args, env_map=None, bg_color=bg_color,mask=dynamic_mask_3d)
            # render_pkg_static = render_pvg(viewpoint_camera, gaussians, args, env_map=None, bg_color=bg_color,mask=knn_mask)
            # render_pkg_bg = render_pvg(viewpoint_camera, gaussians, args, env_map=None, bg_color=bg_color,mask=~(knn_mask|dynamic_mask_3d))
            # image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            # image_dynamic = torch.clamp(render_pkg_dynamic["render"], 0.0, 1.0)
            # image_static = torch.clamp(render_pkg_static["render"], 0.0, 1.0)
            # image_bg = torch.clamp(render_pkg_bg["render"], 0.0, 1.0)
            # grid = make_grid([image, image_dynamic, image_static, image_bg], nrow=4, padding=0)
            # save_path = os.path.join(args.model_path, f"exist_dynamic_render_{viewpoint_camera.colmap_id}_beta.png")
            # save_image(grid, save_path)

            
            log_dict["loss_obj_3d"] = loss_obj_3d.item()
            loss += args.lambda_obj_3d * loss_obj_3d

            # Velocity and beta consistency losses
            if args.lambda_consistency > 0 and args.enable_dynamic and iteration > args.consist_epoch:
                total_v_consist_loss = 0.0
                valid_ids = 0
                
                for id in dynamic_ids:
                    selected_mask = (id_pred_3d_detached.squeeze() == id) & exist_mask  # (N,) select Gaussians for this dynamic object
                    if selected_mask.sum() > 5:  
                        selected_id_positions = gaussians._xyz.squeeze().detach()[selected_mask]  # (C,3) current object's 3D positions
                        selected_id_velocity = velocity[selected_mask]  # (C,3) current object's velocities
                        selected_id_beta = beta[selected_mask]  # (C,1) current object's beta
                        consist_loss = loss_velocity_consistency(
                            selected_id_positions,  # (C,3)
                            selected_id_velocity,  # (C,3)
                            selected_id_beta,  # (C,1)
                            k=5, 
                            alpha=0.4,
                            gamma=0.4,
                        )
                        
                        total_v_consist_loss += consist_loss
                        valid_ids += 1

                
                if valid_ids > 0:
                    avg_consist_loss = total_v_consist_loss / valid_ids
                    log_dict["loss_velocity_consistency_avg"] = avg_consist_loss.item()
                    loss += args.lambda_consistency * avg_consist_loss
                    

        
    extra_render_pkg = {}
    # 2D Gaussian losses
    if visibility_filter.sum() > 0:
        scale = gaussians.get_scaling[visibility_filter] # (N, 3)
        sorted_scale, _ = torch.sort(scale, dim=-1) # (N, 3)
        
        if args.lambda_min_scale > 0:
            min_scale_loss = torch.relu(sorted_scale[..., 0] - args.min_scale) + torch.relu(args.min_scale - sorted_scale[..., 0]) ** 2 # (N)
            log_dict["min_scale"] = min_scale_loss.mean().item()
            loss += args.lambda_min_scale * min_scale_loss.mean()
        
        if args.lambda_max_scale > 0:
            # penalize the maximum scale which is larger than the max_scale
            max_scale_loss = torch.relu(sorted_scale[..., -1] - args.max_scale)
            valid_mask = max_scale_loss > 0
            log_dict["max_scale"] = max_scale_loss[valid_mask].mean().item()
            loss += args.lambda_max_scale * max_scale_loss[valid_mask].mean()

    if iteration >= args.multi_view_weight_from_iter:
        use_virtul_cam = False

        if nearest_cam is not None:
            patch_size = args.multi_view_patch_size
            sample_num = args.multi_view_sample_num
            pixel_noise_th = args.multi_view_pixel_noise_th
            total_patch_size = (patch_size * 2 + 1) ** 2
            ncc_weight = args.multi_view_ncc_weight
            geo_weight = args.multi_view_geo_weight
            ## compute geometry consistency mask and loss
            H, W = render_pkg["depth"].squeeze().shape
            ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
            pixels = (
                torch.stack([ix, iy], dim=-1).float().to(render_pkg["depth"].device)
            )

            nearest_render_pkg = render_pvg(
                nearest_cam,
                gaussians,
                args,
                bg_color,
                env_map=env_map,
                is_training=True,
            )

            pts = gaussians.get_points_from_depth(viewpoint_camera, render_pkg["depth"])
            pts_in_nearest_cam = (
                pts @ nearest_cam.world_view_transform[:3, :3]
                + nearest_cam.world_view_transform[3, :3]
            )
            map_z, d_mask = gaussians.get_points_depth_in_depth_map(
                nearest_cam, nearest_render_pkg["depth"], pts_in_nearest_cam
            )

            pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
            pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
            R = torch.tensor(nearest_cam.R).float().cuda()
            T = torch.tensor(nearest_cam.T).float().cuda()
            pts_ = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
            pts_in_view_cam = (
                pts_ @ viewpoint_camera.world_view_transform[:3, :3]
                + viewpoint_camera.world_view_transform[3, :3]
            )
            pts_projections = torch.stack(
                [
                    pts_in_view_cam[:, 0] * viewpoint_camera.fx / pts_in_view_cam[:, 2]
                    + viewpoint_camera.cx,
                    pts_in_view_cam[:, 1] * viewpoint_camera.fy / pts_in_view_cam[:, 2]
                    + viewpoint_camera.cy,
                ],
                -1,
            ).float()
            pixel_noise = torch.norm(
                pts_projections - pixels.reshape(*pts_projections.shape), dim=-1
            )
            # static_mask = (loss_mult > EPS).reshape(-1)
            static_mask = dynamic_mask.reshape(-1)  # true for static
            
            d_mask = d_mask & (pixel_noise < pixel_noise_th) &  (static_mask) & (alpha_mask.reshape(-1))
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            weights[~d_mask] = 0
            
            extra_render_pkg['rendered_distance'] = render_pkg['rendered_distance']
            extra_render_pkg['d_mask'] = weights.reshape(1, H, W)
            if d_mask.sum() > 0 and geo_weight > 0:
                geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                loss += geo_loss
                if use_virtul_cam is False and ncc_weight > 0:
                    with torch.no_grad():
                        ## sample mask
                        d_mask = d_mask.reshape(-1)
                        valid_indices = torch.arange(
                            d_mask.shape[0], device=d_mask.device
                        )[d_mask]
                        if d_mask.sum() > sample_num:
                            index = np.random.choice(
                                d_mask.sum().cpu().numpy(), sample_num, replace=False
                            )
                            valid_indices = valid_indices[index]

                        weights = weights.reshape(-1)[valid_indices]
                        ## sample ref frame patch
                        pixels = pixels.reshape(-1, 2)[valid_indices]
                        offsets = patch_offsets(patch_size, pixels.device)
                        ori_pixels_patch = (
                            pixels.reshape(-1, 1, 2) / viewpoint_camera.ncc_scale
                            + offsets.float()
                        )

                        H, W = gt_image_gray.squeeze().shape
                        pixels_patch = ori_pixels_patch.clone()
                        pixels_patch[:, :, 0] = (
                            2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                        )
                        pixels_patch[:, :, 1] = (
                            2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                        )
                        ref_gray_val = F.grid_sample(
                            gt_image_gray.unsqueeze(1),
                            pixels_patch.view(1, -1, 1, 2),
                            align_corners=True,
                        )
                        ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                        ref_to_neareast_r = (
                            nearest_cam.world_view_transform[:3, :3].transpose(-1, -2)
                            @ viewpoint_camera.world_view_transform[:3, :3]
                        )
                        ref_to_neareast_t = (
                            -ref_to_neareast_r
                            @ viewpoint_camera.world_view_transform[3, :3]
                            + nearest_cam.world_view_transform[3, :3]
                        )

                    ## compute Homography
                    ref_local_n = render_pkg["normal"].permute(1, 2, 0) # (H, W, 3) [0, 1]
                    ref_local_n = ref_local_n * 2.0 - 1.0
                    ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]

                    ref_local_d = render_pkg['rendered_distance'].squeeze()

                    ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                    H_ref_to_neareast = (
                        ref_to_neareast_r[None]
                        - torch.matmul(
                            ref_to_neareast_t[None, :, None].expand(
                                ref_local_d.shape[0], 3, 1
                            ),
                            ref_local_n[:, :, None]
                            .expand(ref_local_d.shape[0], 3, 1)
                            .permute(0, 2, 1),
                        )
                        / ref_local_d[..., None, None]
                    )
                    H_ref_to_neareast = torch.matmul(
                        nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(
                            ref_local_d.shape[0], 3, 3
                        ),
                        H_ref_to_neareast,
                    )
                    H_ref_to_neareast = H_ref_to_neareast @ viewpoint_camera.get_inv_k(
                        viewpoint_camera.ncc_scale
                    )

                    ## compute neareast frame patch
                    grid = patch_warp(
                        H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch
                    )
                    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                    _, nearest_image_gray = nearest_cam.get_image()
                    sampled_gray_val = F.grid_sample(
                        nearest_image_gray[None],
                        grid.reshape(1, -1, 1, 2),
                        align_corners=True,
                    )
                    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                    ## compute loss
                    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                    mask = ncc_mask.reshape(-1)
                    ncc = ncc.reshape(-1) * weights
                    ncc = ncc[mask].squeeze()

                    if mask.sum() > 0:
                        ncc_loss = ncc_weight * ncc.mean()
                        log_dict["multi_view_ncc"] = ncc_loss.item()
                        loss += ncc_loss

    
    extra_render_pkg["t_map"] = t_map
    extra_render_pkg["v_map"] = v_map
    extra_render_pkg["depth"] = depth
    extra_render_pkg["obj_dc"] = object_dc

    extra_render_pkg["dynamic_mask"] = dynamic_mask
    # extra_render_pkg["dino_cosine"] = dino_part

    log_dict.update(metrics)

    return loss, log_dict, extra_render_pkg


def render_pvg_wrapper(
    args,
    viewpoint_camera: Camera,
    gaussians: DynamicGaussianModel,
    background: torch.Tensor,
    time_interval: float,
    env_map,
    iterations,
    camera_id=None,
    nearest_cam: Camera = None,
    dynamic_dict: dict = None,
):

    # render v and t scale map
    v = gaussians.get_inst_velocity  # average velocity in equation 10
    t_scale = gaussians.get_scaling_t.clamp_max(2) # beta
    other = [t_scale, v]

    if np.random.random() < args.lambda_self_supervision:
        # time_shift = 3 * (np.random.random() - 0.5) * time_interval
        time_shift = 4 * (np.random.random() - 0.5) * time_interval
    else:
        time_shift = None

    render_pkg = render_pvg(
        viewpoint_camera,
        gaussians,
        args,
        background,
        env_map=env_map,
        other=other,
        time_shift=time_shift,
        is_training=True
    )
   
    loss, log_dict, extra_render_pkg = calculate_loss(
        gaussians, viewpoint_camera, background, args, render_pkg, env_map, iterations, camera_id, nearest_cam=nearest_cam, dynamic_dict=dynamic_dict
    )

    render_pkg.update(extra_render_pkg)
    # loss: overall loss
    # log_dict: a dictionary of loss components
    # extra_render_pkg: a dictionary of  "t_map" "v_map" "depth" "dynamic_mask" "dino_cosine"

    return loss, log_dict, render_pkg

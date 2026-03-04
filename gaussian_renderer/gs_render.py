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
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.cameras import Camera
import torch.nn.functional as F
import numpy as np
import kornia
from utils.loss_utils import psnr, ssim, tv_loss, lncc
from utils.graphics_utils import patch_offsets, patch_warp, render_normal
EPS = 1e-5


def render_original_gs(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    env_map=None,
    mask=None,
    is_training=False,
    return_depth_normal=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")+ 0)
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

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None


    if pipe.compute_cov3D_python:
        # false
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
    
    normals = pc.get_normal_v2(viewpoint_camera, means3D) 
    # Transform normals to camera space
    normals = normals @ viewpoint_camera.world_view_transform[:3, :3]

    

    # Prefilter
    if mask is None:
        mask = torch.ones_like(means3D[:, 0], dtype=torch.bool)
    else:
        mask = mask 

    pts_in_cam = (
        means3D @ viewpoint_camera.world_view_transform[:3, :3]
        + viewpoint_camera.world_view_transform[3, :3]
    )

    depth_z = pts_in_cam[:, 2:3]


    local_distance = (normals * pts_in_cam).sum(-1).abs() # (N,3) @ (N,3) -> (N)
    
    depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device) # (N,2)
    depth_alpha[mask] = torch.cat([depth_z, torch.ones_like(depth_z)], dim=1)[mask] 
    
    
    features = torch.cat([depth_alpha, normals, local_distance.unsqueeze(-1)], dim=1) # (N,2+3+1) -> (N,6)
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    _, rendered_image, rendered_feature, radii = rasterizer(
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


    rendered_depth, rendered_opacity, rendered_normal, rendered_distance = (
        rendered_feature.split([1, 1, 3, 1], dim=0)
    )
    # rendered_normal = F.normalize(rendered_normal, dim=0)
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
        "viewspace_points": screenspace_points, # gradients of the 2D (screen-space) means
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        "alpha": rendered_opacity,
        "normal": rendered_normal,
        "scaling": scales,
        "rendered_distance": rendered_distance,
    }

   

    if return_depth_normal:
        depth_normal = (
            render_normal(viewpoint_camera, rendered_depth.squeeze())
            * (rendered_opacity).detach()
        )
        return_dict.update({"depth_normal": depth_normal * 0.5 + 0.5})
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict

def render_combined_gs(
    viewpoint_camera: Camera,
    static_gaussians: GaussianModel,  
    dynamic_gaussians: GaussianModel,  
    pipe,
    bg_color: torch.Tensor,
    env_map=None,
    is_training=False,
):


    static_xyz = static_gaussians.get_xyz.detach()  # (N, 3)
    static_features = static_gaussians.get_features.detach()  # (N, 3, C)
    static_opacity = static_gaussians.get_opacity.detach()  # (N, 1)
    static_scaling = static_gaussians.get_scaling.detach()
    static_rotation = static_gaussians.get_rotation.detach()
    

    dynamic_xyz = dynamic_gaussians.get_xyz
    dynamic_features = dynamic_gaussians.get_features
    dynamic_opacity = dynamic_gaussians.get_opacity*1.5
    dynamic_scaling = dynamic_gaussians.get_scaling
    dynamic_rotation = dynamic_gaussians.get_rotation
    

    combined_xyz = torch.cat([static_xyz, dynamic_xyz], dim=0)
    combined_features = torch.cat([static_features, dynamic_features], dim=0)
    combined_opacity = torch.cat([static_opacity, dynamic_opacity], dim=0)
    combined_scaling = torch.cat([static_scaling, dynamic_scaling], dim=0)
    combined_rotation = torch.cat([static_rotation, dynamic_rotation], dim=0)

    num_static = static_xyz.shape[0]
    num_dynamic = dynamic_xyz.shape[0]
    

    
    # Create zero tensor for screen space points
    
    screenspace_points = (torch.zeros_like(combined_xyz, dtype=combined_xyz.dtype, requires_grad=True, device="cuda")+0)
    try:
        screenspace_points.retain_grad()  # 保留梯度
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
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=dynamic_gaussians.active_sh_degree,  # 使用动态GS的参数
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    

    static_normals = static_gaussians.get_normal_v2(viewpoint_camera, static_xyz).detach()
    dynamic_normals = dynamic_gaussians.get_normal_v2(viewpoint_camera, dynamic_xyz)
    combined_normals = torch.cat([static_normals, dynamic_normals], dim=0)
    combined_normals = combined_normals @ viewpoint_camera.world_view_transform[:3, :3]


    mask = torch.ones_like(combined_xyz[:, 0], dtype=torch.bool)
    

    pts_in_cam = (
        combined_xyz @ viewpoint_camera.world_view_transform[:3, :3]
        + viewpoint_camera.world_view_transform[3, :3]
    )

    depth_z = pts_in_cam[:, 2:3]
    local_distance = (combined_normals * pts_in_cam).sum(-1).abs()
    

    depth_alpha = torch.zeros(combined_xyz.shape[0], 2, dtype=torch.float32, device=combined_xyz.device)
    depth_alpha[mask] = torch.cat([depth_z, torch.ones_like(depth_z)], dim=1)[mask]
    
    features = torch.cat([depth_alpha, combined_normals, local_distance.unsqueeze(-1)], dim=1)
    
    # 渲染
    _, rendered_image, rendered_feature, radii = rasterizer(
        means3D=combined_xyz,
        means2D=screenspace_points,
        shs=combined_features,
        colors_precomp=None,
        features=features,
        opacities=combined_opacity,
        scales=combined_scaling,
        rotations=combined_rotation,
        cov3D_precomp=None,
        mask=mask,
    )

    rendered_depth, rendered_opacity, rendered_normal, rendered_distance = (
        rendered_feature.split([1, 1, 3, 1], dim=0)
    )
    rendered_normal = rendered_normal * 0.5 + 0.5

    rendered_image_before = rendered_image
    if env_map is not None:
        bg_color_from_envmap = env_map(
            viewpoint_camera.get_world_directions(is_training).permute(1, 2, 0)
        ).permute(2, 0, 1)
        rendered_image = rendered_image + (1 - rendered_opacity) * bg_color_from_envmap

    # 重要：只返回动态GS部分的visibility_filter和radii
    # 因为只有动态GS需要进行稠密化等操作
    dynamic_visibility_filter = radii[num_static:] > 0  # 只保留动态部分
    dynamic_radii = radii[num_static:]  # 只保留动态部分
    # dynamic_screenspace_points = screenspace_points[num_static:]  # 只保留动态部分


    return_dict = {
        "render": rendered_image,
        "render_nobg": rendered_image_before,
        "viewspace_points": screenspace_points,  # 只返回动态部分
        "visibility_filter": dynamic_visibility_filter,  # 只返回动态部分
        "radii": dynamic_radii,  # 只返回动态部分
        "depth": rendered_depth,
        "alpha": rendered_opacity,
        "normal": rendered_normal,
        "scaling": combined_scaling[num_static:],  # 只返回动态部分
        "rendered_distance": rendered_distance,
    }


    return return_dict

def calculate_loss(
    gaussians: GaussianModel,
    viewpoint_camera: Camera,
    bg_color: torch.Tensor,
    args,
    render_pkg: dict,
    env_map,
    init: bool = False,
    nearest_cam: Camera = None,
    iteration: int = 0,
):
    log_dict = {}

    image = render_pkg["render"]
    depth = render_pkg["depth"]
    alpha = render_pkg["alpha"]
    visibility_filter = render_pkg["visibility_filter"]
    
    sky_mask = (
        viewpoint_camera.sky_mask.cuda()
        if viewpoint_camera.sky_mask is not None
        else torch.zeros_like(alpha, dtype=torch.bool)
    )

    semantic_mask = (
        viewpoint_camera.semantic_mask.cuda()
        if viewpoint_camera.semantic_mask is not None
        else torch.zeros_like(alpha, dtype=torch.bool)
    )


    sky_depth = 900
    depth = depth / alpha.clamp_min(EPS) #透明度接近0的区域，深度将被设置为无穷大
    
    if env_map is not None:
        # 对深度进行调和平均，透明度低的部分认为有“天空分量”，将增加深度
        # 透明度为0的部分将直接设置为sky_depth
        # 透明度为1的部分将直接使用原始深度
        if args.depth_blend_mode == 0:  # harmonic mean
            depth = 1 / (
                alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth
            ).clamp_min(EPS)
        elif args.depth_blend_mode == 1:
            depth = alpha * depth + (1 - alpha) * sky_depth

    gt_image, gt_image_gray = viewpoint_camera.get_image()
    

    loss = 0
    if init:
        image_for_color_loss = image *(~semantic_mask)  # 只计算非语义区域的颜色损失
        gt_image_for_color_loss = gt_image * (~semantic_mask) 
        
    else:
        image_for_color_loss = image * semantic_mask
        gt_image_for_color_loss = gt_image * semantic_mask
    
    
    loss_l1 = F.l1_loss(image_for_color_loss, gt_image_for_color_loss) 
    log_dict['loss_l1'] = loss_l1.item()
    loss_ssim = 1.0 - ssim(image_for_color_loss, gt_image_for_color_loss)
    log_dict['loss_ssim'] = loss_ssim.item()
    color_loss = (1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim
    
    loss += args.lambda_color * color_loss

    psnr_for_log = psnr(image, gt_image).double()
    log_dict["psnr"] = psnr_for_log

    alpha_mask = alpha.data > EPS  # detached (1, H, W)
    # 深度监督
    if args.lambda_lidar > 0:
        pts_depth = viewpoint_camera.gt_pts_depth.cuda()  
        mask = pts_depth > 0  # 有效的深度
        if init:
            mask = torch.logical_and(mask, ~semantic_mask)  # 只保留非语义区域的有效深度
        else:
            # mask = mask  
            mask = torch.logical_and(mask, semantic_mask)  # 只保留语义区域的有效深度
        
        loss_lidar = torch.abs(
            1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)
        ).mean()
        if args.lidar_decay > 0:
            iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
        else:
            iter_decay = 1
        log_dict["loss_lidar"] = loss_lidar.item()
        loss += iter_decay * args.lambda_lidar * loss_lidar
    
    # 法线监督【关闭】
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

    # 深度平滑损失，增强在纹理平滑区域的深度连续性，同时允许图像边缘处存在较大梯度
    if args.lambda_inv_depth > 0:
        # 天空部分已经填充了sky_depth,不用担心逆深度无穷大
        inverse_depth = 1 / (depth + 1e-5)
        if init:
            inverse_depth = inverse_depth * (~semantic_mask)
        else:
            inverse_depth = inverse_depth * semantic_mask
        
        loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
            inverse_depth[None], gt_image_for_color_loss[None]
        )
        log_dict["loss_inv_depth"] = loss_inv_depth.item()
        loss = loss + args.lambda_inv_depth * loss_inv_depth

    # 迫使天空区域的透明度接近0
    if args.lambda_sky_opa > 0:
        o = alpha.clamp(1e-6, 1 - 1e-6)  
        sky = sky_mask.float() # (1, H, W)
        non_sky = (~sky_mask).float()  # (1, H, W)
        loss_sky_opa = (-sky * torch.log(1 - o)).mean() # + (-non_sky * torch.log(o)).mean()
        semantic_mask = semantic_mask.float()  # (1, H, W)
        if init:
            loss_sky_opa = loss_sky_opa + 0.05 * (-semantic_mask * torch.log(1 - o)).mean() # 语义区域透明度也要接近0
        else:
            loss_sky_opa = loss_sky_opa + 0.05 * (-(1-semantic_mask) * torch.log(1 - o)).mean()
        log_dict["loss_sky_opa"] = loss_sky_opa.item()
        loss = loss + args.lambda_sky_opa * loss_sky_opa
        # print((args.lambda_sky_opa * loss_sky_opa).item())
    

    # 透明度熵损失（要求非天空非语义区域的透明度分布越均匀越好）
    if args.lambda_opacity_entropy > 0:
        o = alpha.clamp(1e-6, 1 - 1e-6)
        loss_opacity_entropy = -(o * torch.log(o)).mean()
        log_dict["loss_opacity_entropy"] = loss_opacity_entropy.item()
        loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

    # 尺度损失（要求尺度分布越均匀越好） 只在3D GS使用
    if args.lambda_scaling > 0 and not init:
        scaling = gaussians.get_scaling[visibility_filter]  # (N, 3)
        scaling_loss = (
            (scaling - scaling.mean(dim=-1, keepdim=True)).abs().sum(-1).mean()
        )

        log_dict["scaling_loss"] = scaling_loss.item()
        loss = loss + args.lambda_scaling * scaling_loss
    
    extra_render_pkg = {}
    extra_render_pkg["t_map"] = torch.zeros_like(alpha)
    extra_render_pkg["v_map"] = torch.zeros_like(alpha)
    extra_render_pkg["depth"] = torch.zeros_like(alpha)
    extra_render_pkg["dynamic_mask"] = torch.zeros_like(alpha)
    extra_render_pkg["dino_cosine"] = torch.zeros_like(alpha)
    return loss, log_dict, extra_render_pkg


def render_gs_origin_wrapper(
    args,
    viewpoint_camera: Camera,
    gaussians: GaussianModel,
    background: torch.Tensor,
    env_map,
    init: bool = False,
    nearest_cam: Camera = None,
    iteration: int = 0,

):

    render_pkg = render_original_gs(
        viewpoint_camera, 
        gaussians, 
        args, 
        background, 
        env_map=env_map, 
    )

    loss, log_dict, extra_render_pkg = calculate_loss(
        gaussians, viewpoint_camera, background, args, render_pkg, env_map, init, nearest_cam=nearest_cam,iteration=iteration
    )


    return loss, log_dict, render_pkg


def render_gs_combined_wrapper(
    args,
    viewpoint_camera: Camera,
    static_gaussians: GaussianModel,  # 静态GS（不优化）
    dynamic_gaussians: GaussianModel,  # 动态GS（需要优化）
    background: torch.Tensor,
    env_map,
    init: bool = False,
    nearest_cam: Camera = None,
    iteration: int = 0,
):

    render_pkg = render_combined_gs(
        viewpoint_camera, 
        static_gaussians, 
        dynamic_gaussians, 
        args, 
        background, 
        env_map=env_map, 
        is_training=True
    )

    loss, log_dict, extra_render_pkg = calculate_loss(
        dynamic_gaussians, viewpoint_camera, background, args, render_pkg, env_map, init, nearest_cam=nearest_cam,iteration=iteration
    )

    return loss, log_dict, render_pkg

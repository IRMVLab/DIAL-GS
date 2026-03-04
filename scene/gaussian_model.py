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
import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_step_lr_func
from torch import nn
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.system_utils import mkdir_p
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import rotation_to_quaternion, quaternion_multiply, quaternion_to_rotation_matrix
import torch.nn.functional as F
from scene.cameras import Camera
import logging



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.scaling_t_activation = torch.exp
        self.scaling_t_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)

    def __init__(self, args):
        self.active_sh_degree = 0 
        self.max_sh_degree = args.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._normal = torch.empty(0)
        
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1
        self.contract = args.contract
        self.big_point_threshold = args.big_point_threshold
        self.isotropic = args.isotropic
        self.random_init_point = args.random_init_point # 20000

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args=None):
        (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.setup_functions()
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.isotropic:
            scaling_out = torch.cat([self._scaling[:, :1, ...], self._scaling[:, :1, ...], self._scaling[:, :1, ...]], dim=1)
            return self.scaling_activation(scaling_out)
        else:
            return self.scaling_activation(self._scaling)


    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_normal(self,c2w=None, mean3d=None, from_scaling=False):
        if not from_scaling:
            return self.normal_activation(self._normal) 
        else:
            assert c2w is not None and mean3d is not None, "c2w and mean3d must be provided if from_scaling is True"
            quats = self.get_rotation # normalized quaternion [N, 4]
            scaling = self.get_scaling # [N, 3]
            normals = F.one_hot(torch.argmin(scaling, dim=-1), num_classes=3).float() # [N, 3] 
            rotation = quaternion_to_rotation_matrix(quats) # [N, 3, 3]
            normals = torch.bmm(rotation, normals.unsqueeze(-1)).squeeze(-1) # [N, 3]
            normals = self.normal_activation(normals) # [N, 3]
            viewdirs = (-mean3d.detach() + c2w[:3, 3].reshape(-1, 3).repeat(mean3d.shape[0], 1).detach()) # [N, 3]
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True) # [N, 3]
            dots = (normals * viewdirs).sum(dim=-1) # [N]
            negative_dot_indices = dots < 0
            normals[negative_dot_indices] = -normals[negative_dot_indices]
            self._normal.data = normals # [N, 3]
            return normals # [N, 3]
        
    def get_normal_v2(self, view_cam, xyz):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_rotation_matrix(self):
        return quaternion_to_rotation_matrix(self.get_rotation)

    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
            
    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1) ** 2

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float,init_opacity=0.5):
        # pts = np.asarray(pcd.points)
        # print("PointCloud min/max: ", pts.min(0)[0], pts.max(0)[0])
        # print("PointCloud mean: ", pts.mean(0))
        self.spatial_lr_scale = spatial_lr_scale
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # fused_point_cloud = torch.cat((self.get_xyz.cuda(),torch.from_numpy(np.asarray(pcd.points)).float().cuda()))



        ## random up and far
        r_max = 100000
        r_min = 2
        num_sph = self.random_init_point # 200000

        theta = 2*torch.pi*torch.rand(num_sph)
        phi = (torch.pi/2*0.99*torch.rand(num_sph))**1.5 # x**a decay
        s = torch.rand(num_sph)
        r_1 = s*1/r_min+(1-s)*1/r_max
        r = 1/r_1
        pts_sph = torch.stack([r*torch.cos(theta)*torch.cos(phi), r*torch.sin(theta)*torch.cos(phi), r*torch.sin(phi)],dim=-1).cuda()

        r_rec = r_min
        num_rec = self.random_init_point
        pts_rec = torch.stack([r_rec*(torch.rand(num_rec)-0.5),r_rec*(torch.rand(num_rec)-0.5),
                            r_rec*(torch.rand(num_rec))],dim=-1).cuda()

        pts_sph = torch.cat([pts_rec, pts_sph], dim=0)
        pts_sph[:,2] = -pts_sph[:,2]+1 

        fused_point_cloud = torch.cat([fused_point_cloud, pts_sph], dim=0) # [N, 3]
        features = torch.cat([features,
                            torch.zeros([pts_sph.size(0), features.size(1), features.size(2)]).float().cuda()],
                            dim=0)
        

        logging.info("Number of points at initialization: {}".format(fused_point_cloud.shape[0]))
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001) #输出形状为 (N,) 的张量，表示每个点到其最近邻点的平方距离。
        dist2 = dist2[self._xyz.shape[0]:]
        # scales = torch.sqrt(dist2)
        # print("Avg scales: ", scales.mean().item())
        # print("Max scales: ", scales.max().item())
        # print("Min scales: ", scales.min().item())
        # print("Scales std: ", scales.std().item())
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)


        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        # opacities = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        opacities = inverse_sigmoid(init_opacity*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # 0.01

        normal = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        normal[..., 2] = 1.0

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def create_from_pcd_merge(self, pcd_A: BasicPointCloud, pcd_B: BasicPointCloud, spatial_lr_scale: float, init_opacity=0.5):
        """
        从两个点云对象创建高斯模型：
        - pcd_A: 用于GS初始化的点云（实际添加到模型中）
        - pcd_B: 已有的固定点云，仅用于计算A点云初始化时的GS尺寸
        
        Args:
            pcd_A (BasicPointCloud): 用于GS初始化的点云数据
            pcd_B (BasicPointCloud): 已有的固定点云，用于计算尺寸
            spatial_lr_scale (float): 空间学习率缩放因子
            init_opacity (float): 初始透明度值，默认0.5
        """
        self.spatial_lr_scale = spatial_lr_scale
        
        # 处理点云A的颜色和特征
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd_A.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        
        # 获取点云A的坐标
        point_cloud_A = torch.tensor(np.asarray(pcd_A.points)).float().cuda()
        
        logging.info("Number of points at initialization: {}".format(point_cloud_A.shape[0]))
        
        # 关键部分：使用点云A和点云B的合并来计算距离，但只为点云A（和随机点）计算尺寸
        if pcd_B.points.shape[0] > 0:
            # 将点云B转换为CUDA张量
            point_cloud_B = torch.tensor(np.asarray(pcd_B.points)).float().cuda()
            
            # 合并点云A、随机点和点云B用于距离计算
            combined_for_distance = torch.cat([point_cloud_A, point_cloud_B], dim=0)
            
            # 计算距离（包含所有点的KNN）
            dist2_all = torch.clamp_min(distCUDA2(combined_for_distance), 0.0000001)
            
            # 只保留fused_point_cloud对应的距离（前N个点）
            dist2 = dist2_all[:point_cloud_A.shape[0]]
        else:
            # 如果点云B为空，则只使用fused_point_cloud计算距离
            dist2 = torch.clamp_min(distCUDA2(point_cloud_A), 0.0000001)
        
        # 计算尺寸参数
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # 初始化旋转（单位四元数）
        rots = torch.zeros((point_cloud_A.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        # 初始化透明度
        opacities = inverse_sigmoid(init_opacity*torch.ones((point_cloud_A.shape[0], 1), dtype=torch.float, device="cuda"))

        # 初始化法线
        normal = torch.zeros((point_cloud_A.shape[0], 3), dtype=torch.float, device="cuda")
        normal[..., 2] = 1.0

        # 设置模型参数
        self._xyz = nn.Parameter(point_cloud_A.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        logging.info(f"Initialized GS with {point_cloud_A.shape[0]} points")
        logging.info(f"Points from pcd_A: {point_cloud_A.shape[0]}")
        logging.info(f"Points from pcd_B used for distance calculation: {pcd_B.points.shape[0]}")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
      
        # 这一步把l绑定到optimizer上，optimizer的param_groups就是l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15) # lr=0表示不使用全局学习率，eps表示adam的epsilon
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=30000)

        
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            state = self.optimizer.state[param]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(param)
            if "step" not in state:
                state["step"] = torch.tensor(0, dtype=torch.long, device=param.device)
        
        # print("When traning setup, the optimizer state is:")
        # for i in range(len(l)):
        #     print(f"Optimizer state for {l[i]['name']}: {self.optimizer.state[l[i]['params'][0]].keys() if self.optimizer.state[l[i]['params'][0]] is not None else None}")

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

    def reset_opacity(self):
        # 将透明度大于0.01的GS重置为0.01
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        # 把指定名称的参数替换为新的值并重置优化器状态
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None) # 从优化器获取该参数的状态
                stored_state["exp_avg"] = torch.zeros_like(tensor) # 重置改参数的动量
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor) # 重置改参数的二次动量

                del self.optimizer.state[group['params'][0]] # 删除旧的参数状态
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True)) # 将新的张量替换旧的参数
                self.optimizer.state[group['params'][0]] = stored_state # 将新的参数状态添加到优化器中
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        # 根据mask来裁剪优化器中的参数及其状态
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))) # 更新参数
                self.optimizer.state[group['params'][0]] = stored_state # 更新优化器状态
                optimizable_tensors[group["name"]] = group["params"][0] 
            else:
                print("When pruning, the optimizer state is None")
                exit(0)
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        
        # 更新优化器中的参数和状态
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        # 更新对象属性（而非更新优化器中的参数）
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._normal = optimizable_tensors["normal"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        # 扩展参数、状态
        optimizable_tensors = {} 
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1 # 每个参数组只能有一个参数
            extension_tensor = tensors_dict[group["name"]] # 获取要扩展的张量
            stored_state = self.optimizer.state.get(group['params'][0], None) # 获取参数状态
            if stored_state is not None:
                # 如果当前参数组中，参数的状态不为空，则扩展动量和二次动量
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),dim=0)
                # 删除旧的参数状态
                del self.optimizer.state[group['params'][0]]
                
                # 扩展参数
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                # 扩展状态
                self.optimizer.state[group['params'][0]] = stored_state
                # 存入字典（为了后续更新对象属性）
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 若参数状态为空，直接创建新的状态
                stored_state = {}
                stored_state["exp_avg"] = torch.zeros_like(group["params"][0])
                stored_state["exp_avg_sq"] = torch.zeros_like(group["params"][0])
                
                # 扩展参数
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0].cuda(), extension_tensor), dim=0).requires_grad_(True))
                # 扩展状态
                self.optimizer.state[group['params'][0]] = stored_state
                # 存入字典（为了后续更新对象属性）
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_normal, new_scaling,
                              new_rotation):
        
        # 扩展参数、清空梯度积累
        d = {"xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "normal": new_normal,
            "scaling": new_scaling,
            "rotation": new_rotation,
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d) # 扩展参数、状态
        # 更新对象属性
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._normal = optimizable_tensors["normal"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # 更新梯度积累、清零最大2D半径
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # grads 是一个形状为 (N, 3) 的张量，表示每个点的平均积累梯度
        # grad_threshold 是一个标量，表示梯度的阈值
        # N 是一个整数，表示稠密化的倍数
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        # 梯度筛选（梯度较大的点影响力更强，需要稠密化）
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) # (M,)

        # 计算缩放因子，用于控制稠密化的程度
        if self.contract:
            scale_factor = self._xyz.norm(dim=-1)*scene_extent-1 # -0
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/scene_extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]/scene_extent #(N,)
        
        # 尺度筛选（点的尺寸最大值（激活后）大于percent_dense * scene_extent * scale_factor才能稠密化）
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,dim=1).values > self.percent_dense * scene_extent * scale_factor)
        # (M,)
        
        print("\n \n 分裂稠密化点数：", selected_pts_mask.sum().item())
        
        decay_factor = N*0.8 # 分裂后，尺度变为原来的1/(0.8*N)
        

        # 复制选中的点（稠密化）
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (decay_factor)) # 分裂后尺寸除以N*0.8倍
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1) # (M*N, 3) M是选中的点的数量，N是稠密化的倍数
        means = torch.zeros((stds.size(0), 3), device="cuda") # (M*N, 3) 均值为0
        samples = torch.normal(mean=means, std=stds)    # (M*N, 3) 对应M*N个三维正态分布，每个分布的标准差与scale相同
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1) # (M*N, 3, 3) 复制选中的点的旋转矩阵
        xyz = self.get_xyz[selected_pts_mask] # (M, 3) 选中的点的坐标

        # 新点坐标是在原来坐标上向主轴方向进行偏移
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz.repeat(N, 1)

        # 尺寸过小的点需要恢复尺度（冗余步骤，保证安全）
        not_split_xyz_mask =  torch.max(self.get_scaling[selected_pts_mask], dim=1).values < \
                                self.percent_dense * scene_extent*scale_factor[selected_pts_mask] #(M,)
        
        new_scaling[not_split_xyz_mask.repeat(N)] = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1))[not_split_xyz_mask.repeat(N)]

        # 扩展参数、状态，更新对象属性
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_normal, new_scaling, new_rotation)
        
        # 删除原始点
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        
        if self.contract:
            scale_factor = self._xyz.norm(dim=-1)*scene_extent-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/scene_extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]/scene_extent

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) 
        # 尺度最大值（激活后）小于percent_dense * scene_extent * scale_factor需要克隆
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling,dim=1).values <= self.percent_dense * scene_extent*scale_factor)
        print("克隆稠密化点数：", selected_pts_mask.sum().item())
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_normal, new_scaling, new_rotation)

    def densify(self, max_grad, extent, init=False):

        # if init:
        #     # 第一阶段，第一次剪枝时将把从未见过的点全部剪枝
        #     invisible_mask = (self.denom ==0).squeeze()
        #     if torch.any(invisible_mask):
        #         print(f"Pruning {torch.sum(invisible_mask)} invisible points")
        #         self.prune_points(invisible_mask)
        
        grads = self.xyz_gradient_accum / self.denom
        invalid_mask = torch.isnan(grads) | torch.isinf(grads) #| (self.denom < 10) 
        if torch.any(invalid_mask):
            print(f"Invalid gradients found: {torch.sum(invalid_mask)}")
            grads[invalid_mask] = 0.0
        
        grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        grads = torch.clamp(grads, 0.0, 1.0)  # 注意这里限制了最大梯度为1.0
        # print(f"Points above threshold: {torch.sum(grads.squeeze() >= max_grad)}")
        # print(f"Max gradient: {grads.max():.6f}")
        # print(f"Mean gradient: {grads.mean():.6f}")
        # print(f"Min denom: {self.denom.min():.1f}")
        # print(f"Max denom: {self.denom.max():.1f}")

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        torch.cuda.empty_cache()

    def prune(self, extent, max_screen_size=None,min_opacity=0.1):
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self.contract:
            scale_factor = self._xyz.norm(dim=-1)*extent-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/extent
            # 距离中心越远，所容许的最大尺度越大
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]/extent
            # 使用统一的尺度因子

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > self.big_point_threshold * extent * scale_factor  ## ori 0.1
            big_prune_mask = torch.logical_or(big_points_vs, big_points_ws)
            print("大点剪枝数目: ", big_prune_mask.sum().item())
            prune_mask = torch.logical_or(prune_mask, big_prune_mask)
        
        print("Pruning points: ", prune_mask.sum().item())
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 更新xyz梯度，用于稠密化判断
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # normals = self._normal.detach().cpu().numpy()
        normals =np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        # activated_scale = torch.max(self.get_scaling, dim=1)[0].detach().cpu().numpy()
        # print(activated_scale.shape)
        
        mask = (self.get_opacity[:, 0].detach().cpu().numpy() > 0) 
        xyz = xyz[mask]
        normals = normals[mask]
        f_dc = f_dc[mask]
        f_rest = f_rest[mask]
        opacities = opacities[mask]
        scale = scale[mask]
        rotation = rotation[mask]
        print("Saving {} points to {}".format(xyz.shape[0], path))

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def add_points_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float,init_opacity: float = 0.5):

        """
        从 BasicPointCloud 对象中添加点云数据到现有的高斯模型。

        Args:
            pcd (BasicPointCloud): 包含点云数据的对象，包含点的坐标、颜色和时间信息。
            spatial_lr_scale (float): 空间学习率缩放因子，用于调整学习率。
        """
        # 设置空间学习率缩放因子
        self.spatial_lr_scale = spatial_lr_scale

        # 提取点云的坐标并转换为 CUDA 张量
        new_points = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if new_points.shape[0] == 0:
            print("No points to add")
            return

        # 提取点云的颜色并转换为球谐函数（SH）表示
        new_colors = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # 初始化特征张量，包含球谐函数的 DC 和其他通道
        new_features = torch.zeros((new_colors.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        new_features[:, :3, 0] = new_colors  # DC 通道
        new_features[:, 3:, 1:] = 0.0  # 其他通道初始化为 0

        point_cloud_A = torch.cat((self.get_xyz.cuda(),torch.from_numpy(np.asarray(pcd.points)).float().cuda()))
        # 计算每个点的最近邻距离平方，并用作尺度初始化
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)  # 最近邻距离平方
        dist2 = dist2[self._xyz.shape[0]:]  # 只保留新点的距离平方
        new_scaling = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)  # 初始化尺度
        # 初始化旋转矩阵为单位四元数
        new_rotations = torch.zeros((new_points.shape[0], 4), device="cuda")
        new_rotations[:, 0] = 1
        new_opacities = inverse_sigmoid(init_opacity* torch.ones((new_points.shape[0], 1), dtype=torch.float, device="cuda"))

        # 初始化法线为 [0, 0, 1]
        new_normals = torch.zeros((new_points.shape[0], 3), dtype=torch.float, device="cuda")
        new_normals[..., 2] = 1.0

       
        # 调用 densification_postfix 方法，将新点添加到高斯模型中
        self.densification_postfix(
            new_xyz=new_points,
            new_features_dc=new_features[:, :, 0:1].transpose(1, 2).contiguous(),
            new_features_rest=new_features[:, :, 1:].transpose(1, 2).contiguous(),
            new_opacities=new_opacities,
            new_normal=new_normals,
            new_scaling=new_scaling,
            new_rotation=new_rotations
        )
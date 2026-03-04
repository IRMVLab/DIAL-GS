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

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, s=64.0, m=0.5):
        """
        ArcFace 实现
        Args:
            num_classes (int): 类别数
            feat_dim (int): 特征维度
            s (float): 缩放因子 (默认 64)
            m (float): 角度 margin (默认 0.5)
        """
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.s = s
        self.m = m

        # 分类层权重
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

        # 预计算 margin 的 cos 和 sin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def classifier(self, features):
        """
        分类器：输入特征 -> 输出 logits（未加 margin）
        Args:
            features (Tensor): [feat_dim, H, W]
        Returns:
            logits (Tensor): [num_classes, H, W]
        """
        # 1. 归一化特征和权重

        height, weight = features.shape[1], features.shape[2]

        features = features.permute(1, 2, 0).contiguous().view(-1, self.feat_dim)  # [H*W, feat_dim]
        features = F.normalize(features, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1) # [num_classes, feat_dim]

        # 2. 计算 cos(theta)
        cos_theta = torch.matmul(features, W.t())  # [H*W, num_classes]
        cos_theta = cos_theta.clamp(-1, 1)
        logits = self.s * cos_theta  # 直接缩放
        logits = logits.view(height, weight, -1).permute(2, 0, 1).contiguous()  # [num_classes, H, W]
       
        return logits

    def forward(self, features, labels):
        """
        训练前向传播
        Args:
            features (Tensor): [batch, feat_dim]
            labels (Tensor): [batch]
        Returns:
            loss (Tensor)
        """
        # 1. 归一化
        features = F.normalize(features, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)

        # 2. cos(theta)
        cos_theta = torch.matmul(features, W.t())
        cos_theta = cos_theta.clamp(-1, 1)

        # 3. sin(theta)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

        # 4. cos(theta + m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        # 5. 生成 logits
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        logits *= self.s

        # 6. CE loss
        loss = F.cross_entropy(logits, labels)
        return loss
    

class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Loss
    自动在一个 batch 内挖 hardest positive 和 hardest negative
    
    Args:
        margin: Triplet loss margin
        p: 距离度量 (默认 L2 欧氏距离)
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, margin=0.3, p=2, reduction="mean"):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (N, D) 特征向量
            labels: (N,) 样本标签
        Returns:
            loss: 标量 loss
        """
        # L2 normalize，保证数值稳定
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算 pairwise 距离矩阵 (N, N)
        dist_matrix = torch.cdist(embeddings, embeddings, p=self.p)

        # 构造标签匹配矩阵 (N, N)，同类=1，不同类=0
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        # hardest positive: 同类中最远的
        pos_mask = label_matrix.float()
        pos_mask.fill_diagonal_(0)  # 自身不算
        hardest_pos_dist = (dist_matrix * pos_mask).max(dim=1)[0]

        # hardest negative: 异类中最近的
        neg_mask = (~label_matrix).float()
        hardest_neg_dist = (dist_matrix + 1e6 * (1 - neg_mask)).min(dim=1)[0]

        # Triplet Loss: [d_pos - d_neg + margin]+
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


    

class DynamicGaussianModel:

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
        self._objects_dc = torch.empty(0) # ID embedding
        self.num_objects = args.embedding_dim # ID embedding dimension
        self.num_classes = 21 # Max IDs, last dimension represents static ID
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._t = torch.empty(0) # new tau
        self._scaling_t = torch.empty(0) # new beta
        self._velocity = torch.empty(0) # new v
        self._normal = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.t_gradient_accum = torch.empty(0) # new
        self.denom = torch.empty(0)
        
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.time_duration = args.time_duration
        self.no_time_split = args.no_time_split
        self.t_grad = args.t_grad
        self.contract = args.contract
        self.t_init = args.t_init
        self.big_point_threshold = args.big_point_threshold
        self.isotropic = args.isotropic

        self.T = args.cycle # 0.2 = l
        self.velocity_decay = args.velocity_decay # 1.0
        self.random_init_point = args.random_init_point # 200000
        self.enable_dynamic = args.enable_dynamic
        self.setup_functions()

        self.classifier = torch.nn.Conv2d(self.num_objects, self.num_classes, kernel_size=1) # input: (8, H, W), output: (21, H, W)
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=5e-4)
        self.classifier.to("cuda")
        # self.arcface_loss = ArcFaceLoss(num_classes=self.num_classes, feat_dim=self.num_objects, s=40, m=0.5).cuda()
        # self.classifier = self.arcface_loss.classifier
        # self.cls_optimizer = torch.optim.Adam(self.arcface_loss.parameters(), lr=5e-4)
        # self.triplet_loss = BatchHardTripletLoss(margin=1.0)
        # self.init_prune=False

    @torch.no_grad()
    def set_transform(self, rotation=None, center=None, scale=None, offset=None, transform=None):
        if transform is not None:
            scale = transform[:3, :3].norm(dim=-1)
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
            xyz_homo = torch.cat([self._xyz.data, torch.ones_like(self._xyz[:, :1])], dim=-1)
            self._xyz.data = (xyz_homo @ transform.T)[:, :3]
            rotation = transform[:3, :3] / scale[:, None]
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
            return

        if center is not None:
            self._xyz.data = self._xyz.data - center
        if rotation is not None:
            self._xyz.data = (self._xyz.data @ rotation.T)
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
        if scale is not None:
            self._xyz.data = self._xyz.data * scale
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
        if offset is not None:
            self._xyz.data = self._xyz.data + offset


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._objects_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self._t,
            self._scaling_t,
            self._velocity,
            self._normal,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.t_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.T,
            self.velocity_decay,
            self.classifier.state_dict(),
            self.cls_optimizer.state_dict()
        )

    def restore(self, model_args, training_args=None):
        (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._objects_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self._t,
            self._scaling_t,
            self._velocity,
            self._normal,
            self.max_radii2D,
            xyz_gradient_accum,
            t_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.T,
            self.velocity_decay,
            classifier_state_dict,
            cls_optmizer_state_dict
        ) = model_args
        self.setup_functions()
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
            self.classifier.load_state_dict(classifier_state_dict)
            self.cls_optimizer.load_state_dict(cls_optmizer_state_dict)
            print("Successfully restore the classifier!")

    @property
    def get_scaling(self):
        if self.isotropic:
            scaling_out = torch.cat([self._scaling[:, :1, ...], self._scaling[:, :1, ...], self._scaling[:, :1, ...]], dim=1)
            return self.scaling_activation(scaling_out)
        else:
            return self.scaling_activation(self._scaling)

    @property
    def get_scaling_t(self):
        if self.enable_dynamic:
            return self.scaling_t_activation(self._scaling_t)
        else:
            return torch.ones_like(self._scaling_t) * 1e10

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_xyz_SHM(self, t):
        if self.enable_dynamic:
            a = 1/self.T * np.pi * 2
            return self._xyz + self._velocity*torch.sin((t-self._t)*a)/a
        else:
            return self._xyz

    @property
    def get_inst_velocity(self):
        if self.enable_dynamic:
            return self._velocity*torch.exp(-self.get_scaling_t/self.T/2*self.velocity_decay)
        else:
            return torch.zeros_like(self._velocity)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_obj_dc(self):
        if self._objects_dc is None or self._objects_dc.shape[0] == 0:
            return None
        else:
            return self._objects_dc # 

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

    def get_marginal_t(self, timestamp):
        if self.enable_dynamic:
            return torch.exp(-0.5 * (self.get_t - timestamp) ** 2 / self.get_scaling_t ** 2)
        else:
            return torch.ones_like(self.get_t)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

       
        
        ## random up and far
        r_max = 100000
        r_min = 2
        num_sph = self.random_init_point

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

        fused_point_cloud = torch.cat([fused_point_cloud, pts_sph], dim=0)
        features = torch.cat([features,
                              torch.zeros([pts_sph.size(0), features.size(1), features.size(2)]).float().cuda()],
                             dim=0)
        
        fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0],self.num_objects), device="cuda")) # (N,16)
        fused_objects = fused_objects[:,:,None] # (N,16,1)

        if pcd.time is None or pcd.time.shape[0] != fused_point_cloud.shape[0]:
            if pcd.time is None:
                time = (np.random.rand(pcd.points.shape[0], 1) * 1.2 - 0.1) * (
                        self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                time = pcd.time

            if self.t_init < 1:
                random_times = (torch.rand(fused_point_cloud.shape[0]-pcd.points.shape[0], 1, device="cuda") * 1.2 - 0.1) * (
                        self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
                pts_times = torch.from_numpy(time.copy()).float().cuda()
                fused_times = torch.cat([pts_times, random_times], dim=0)
            else:
                fused_times = torch.full_like(fused_point_cloud[..., :1],
                                                0.5 * (self.time_duration[1] + self.time_duration[0]))
        else:
            fused_times = torch.from_numpy(np.asarray(pcd.time.copy())).cuda().float()
            fused_times_sh = torch.full_like(pts_sph[..., :1], 0.5 * (self.time_duration[1] + self.time_duration[0]))
            fused_times = torch.cat([fused_times, fused_times_sh], dim=0)

        logging.info("Number of points at initialization: {}".format(fused_point_cloud.shape[0]))

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        dist_t = torch.full_like(fused_times, (self.time_duration[1] - self.time_duration[0])*self.t_init)
        scales_t = self.scaling_t_inverse_activation(torch.sqrt(dist_t))
        velocity = torch.full((fused_point_cloud.shape[0], 3), 0., device="cuda")
        
        opacities = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        normal = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        normal[..., 2] = 1.0

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) 
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))  # # (N,3,1) -> (N,1,3)
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True)) # (N,1,16)
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.enable_dynamic:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            self._velocity = nn.Parameter(velocity.requires_grad_(True))
        else:
            self._t = torch.zeros_like(fused_times)
            self._scaling_t = torch.ones_like(scales_t) * 1e10
            self._velocity = torch.zeros_like(velocity)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if self.enable_dynamic:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._objects_dc], 'lr': training_args.feature_lr / 4, "name": "obj_dc"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self._t], 'lr': training_args.t_lr_init, "name": "t"},
                {'params': [self._scaling_t], 'lr': training_args.scaling_t_lr, "name": "scaling_t"},
                {'params': [self._velocity], 'lr': training_args.velocity_lr * self.spatial_lr_scale, "name": "velocity"},
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            ]
       

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

        final_decay = training_args.position_lr_final / training_args.position_lr_init

        self.t_scheduler_args = get_expon_lr_func(lr_init=training_args.t_lr_init,
                                                    lr_final=training_args.t_lr_init * final_decay,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "t":
                lr = self.t_scheduler_args(iteration)
                param_group['lr'] = lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._objects_dc = optimizable_tensors["obj_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._normal = optimizable_tensors["normal"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.enable_dynamic:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            self._velocity = optimizable_tensors['velocity']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]
        else:
            self._t = torch.zeros_like(self._opacity)
            self._scaling_t = torch.zeros_like(self._opacity)
            self._velocity = torch.zeros_like(self._xyz)
            self.t_gradient_accum = torch.zeros_like(self._opacity)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_obj_dc, new_opacities, new_normal, new_scaling,
                              new_rotation, new_t, new_scaling_t, new_velocity):
        if self.enable_dynamic:
            d = {"xyz": new_xyz,
                "f_dc": new_features_dc,
                "f_rest": new_features_rest,
                "obj_dc": new_obj_dc,
                "opacity": new_opacities,
                "normal": new_normal,
                "scaling": new_scaling,
                "rotation": new_rotation,
                "t": new_t,
                "scaling_t": new_scaling_t,
                "velocity": new_velocity,
                }
        else:
            d = {"xyz": new_xyz,
                "f_dc": new_features_dc,
                "f_rest": new_features_rest,
                "opacity": new_opacities,
                "normal": new_normal,
                "scaling": new_scaling,
                "rotation": new_rotation,
                }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._objects_dc = optimizable_tensors["obj_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._normal = optimizable_tensors["normal"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        if self.enable_dynamic:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            self._velocity = optimizable_tensors['velocity']
        else:
            self._t = torch.zeros_like(self._opacity)
            self._scaling_t = torch.zeros_like(self._opacity)
            self._velocity = torch.zeros_like(self._xyz)
            
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2, time_split=False,
                          joint_sample=True):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        if self.contract:
            scale_factor = self._xyz.norm(dim=-1)*scene_extent-1 # -0
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/scene_extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]/scene_extent

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent*scale_factor)
        decay_factor = N*0.8
        if not self.no_time_split:
            N = N+1

        if time_split:
            padded_grad_t = torch.zeros((n_init_points), device="cuda")
            padded_grad_t[:grads_t.shape[0]] = grads_t.squeeze()
            selected_time_mask = torch.where(padded_grad_t >= grad_t_threshold, True, False)
            extend_thresh = self.percent_dense

            selected_time_mask = torch.logical_and(selected_time_mask,
                                                   torch.max(self.get_scaling_t, dim=1).values > extend_thresh)
            if joint_sample:
                selected_pts_mask = torch.logical_or(selected_pts_mask, selected_time_mask)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (decay_factor))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_obj_dc = self._objects_dc[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        xyz = self.get_xyz[selected_pts_mask]

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz.repeat(N, 1)

        new_t = None
        new_scaling_t = None
        new_velocity = None
        stds_t = self.get_scaling_t[selected_pts_mask].repeat(N, 1)
        means_t = torch.zeros((stds_t.size(0), 1), device="cuda")
        samples_t = torch.normal(mean=means_t, std=stds_t)
        new_t = samples_t+self.get_t[selected_pts_mask].repeat(N, 1)

        new_scaling_t = self.scaling_t_inverse_activation(
            self.get_scaling_t[selected_pts_mask].repeat(N, 1)/ (decay_factor))



        new_velocity = self._velocity[selected_pts_mask].repeat(N, 1)

        new_xyz = new_xyz + self.get_inst_velocity[selected_pts_mask].repeat(N, 1) * (samples_t)

        not_split_xyz_mask =  torch.max(self.get_scaling[selected_pts_mask], dim=1).values < \
                                self.percent_dense * scene_extent*scale_factor[selected_pts_mask]
        new_scaling[not_split_xyz_mask.repeat(N)] = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1))[not_split_xyz_mask.repeat(N)]

        if time_split:
            not_split_t_mask = self.get_scaling_t[selected_pts_mask].squeeze() < extend_thresh
            new_scaling_t[not_split_t_mask.repeat(N)] = self.scaling_t_inverse_activation(
                self.get_scaling_t[selected_pts_mask].repeat(N, 1))[not_split_t_mask.repeat(N)]

        if self.no_time_split:
            new_scaling_t = self.scaling_t_inverse_activation(
                self.get_scaling_t[selected_pts_mask].repeat(N, 1))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_obj_dc, new_opacity, new_normal, new_scaling, new_rotation,
                                   new_t, new_scaling_t, new_velocity)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, time_clone=False):
        t_scale_factor=self.get_scaling_t.clamp(0,self.T)
        t_scale_factor=torch.exp(-t_scale_factor/self.T).squeeze()

        if self.contract:
            scale_factor = self._xyz.norm(dim=-1)*scene_extent-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/scene_extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]/scene_extent

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling,dim=1).values <= self.percent_dense * scene_extent*scale_factor)
        if time_clone:
            selected_time_mask = torch.where(torch.norm(grads_t, dim=-1) >= grad_t_threshold, True, False)
            extend_thresh = self.percent_dense
            selected_time_mask = torch.logical_and(selected_time_mask,
                                                   torch.max(self.get_scaling_t, dim=1).values <= extend_thresh)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_time_mask)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_obj_dc = self._objects_dc[selected_pts_mask]    
        new_opacities = self._opacity[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_velocity = None
        if self.enable_dynamic:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            new_velocity = self._velocity[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_obj_dc, new_opacities, new_normal, new_scaling,
                                   new_rotation, new_t, new_scaling_t, new_velocity)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False):
        if not prune_only:
            
            # if self.init_prune is False:
            #     invisible_mask = (self.denom ==0).squeeze()
            #     if torch.any(invisible_mask):
            #         print(f"Pruning {torch.sum(invisible_mask)} invisible points")
            #         self.prune_points(invisible_mask)
            #     self.init_prune = True


            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            grads_t = self.t_gradient_accum / self.denom
            grads_t[grads_t.isnan()] = 0.0

            if self.t_grad:
                self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, time_clone=True)
                self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t, time_split=True)
            else:
                self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, time_clone=False)
                self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t, time_split=False)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if self.contract:
            scale_factor = self._xyz.norm(dim=-1)*extent-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]/extent

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > self.big_point_threshold * extent * scale_factor  ## ori 0.1
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
        if self.enable_dynamic:
            self.t_gradient_accum[update_filter] += self._t.grad.clone()[update_filter]

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        # l.append('obj_dc')
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
        obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() # (N,16,1)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        # activated_scale = torch.max(self.get_scaling, dim=1)[0].detach().cpu().numpy()
        # print(activated_scale.shape)
        
        mask = (self.get_opacity[:, 0].detach().cpu().numpy() > 0.1) 
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
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, obj_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def save_ply_at_t(self, path, time=0.0, dynamic_mask=None):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz_SHM(time).detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        # normals =np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        marginal_t = self.get_marginal_t(time)
        opacities = self.get_opacity * marginal_t
        
        scale = self._scaling.detach().cpu().numpy() # (N, 3) unactivated before exp
        rotation = self._rotation.detach().cpu().numpy()
        # remove points with low opacity, otherwise there will be too many white ellipsoids
        mask = (marginal_t[:, 0].detach().cpu().numpy() > 0.05) & (self.get_opacity[:, 0].detach().cpu().numpy() > 0.01) 
        if dynamic_mask is not None:
            mask = mask & dynamic_mask

        # the saved format of opacity has to be pre-activated
        opacities = self.inverse_opacity_activation(opacities).detach().cpu().numpy()
        xyz = xyz[mask]
        normals = normals[mask]
        f_dc = f_dc[mask]
        f_rest = f_rest[mask]
        opacities = opacities[mask]
        scale = scale[mask]
        rotation = rotation[mask]
        
        # print("Scale: ", scale.shape)
        # np.save("debug/scale.npy", scale)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        print("Saving {} points".format(xyz.shape[0]))
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def get_points_depth_in_depth_map(self, fov_camera : Camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0) # 0
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.fx / points_in_camera_space[:,2] + fov_camera.cx,
                         points_in_camera_space[:,1] * fov_camera.fy / points_in_camera_space[:,2] + fov_camera.cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera : Camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        return pts
    


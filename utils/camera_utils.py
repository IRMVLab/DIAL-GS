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
import cv2
from scene.cameras import Camera
import numpy as np
from scene.scene_utils import CameraInfo
from tqdm import tqdm
from .graphics_utils import fov2focal


def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
    # resolution_scale [1, 2, 4, 8, 16]
    orig_w, orig_h = cam_info.width, cam_info.height  # 加载后的图像大小，不是原始图像大小


    # args.resolution默认为-1
    
    if args.resolution in [1, 2, 3, 4, 8, 16, 32]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            global_down = 1 # default
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale)) # (W, H)


    # resolution_scale 是某个训练阶段的缩放比例
    # args.resolution 是用户配置的全局缩放比例
    # scale 是目前的缩放比例
    # resolution 是当前的分辨率（基于加载的图像大小）

    if cam_info.cx:
        cx = cam_info.cx / scale
        cy = cam_info.cy / scale
        fy = cam_info.fy / scale
        fx = cam_info.fx / scale
    else:
        cx = None
        cy = None
        fy = None
        fx = None
    

    if cam_info.image.shape[:2] != resolution[::-1]:
        image_rgb = cv2.resize(cam_info.image, resolution)
    else:
        image_rgb = cam_info.image
    
    image_rgb = torch.from_numpy(image_rgb).float().permute(2, 0, 1) # H, W, 3 -> 3, H, W
    gt_image = image_rgb[:3, ...] # 只保留前3个通道，去掉alpha通道（虽然并没有）

    if cam_info.sky_mask is not None:
        if cam_info.sky_mask.shape[:2] != resolution[::-1]:
            sky_mask = cv2.resize(cam_info.sky_mask, resolution)
            # print("对sky_mask进行双线性插值")
        else:
            sky_mask = cam_info.sky_mask
        if len(sky_mask.shape) == 2:
            sky_mask = sky_mask[..., None]
        sky_mask = torch.from_numpy(sky_mask).float().permute(2, 0, 1) # H, W, 1 -> 1, H, W
    else:
        sky_mask = None

    if cam_info.semantic_mask is not None:
        if cam_info.semantic_mask.shape[:2] != resolution[::-1]:
            semantic_mask = cv2.resize(cam_info.semantic_mask, resolution)
            # print("对semantic_mask进行双线性插值")
        else:
            semantic_mask = cam_info.semantic_mask
        if len(semantic_mask.shape) == 2:
            semantic_mask = semantic_mask[..., None]
        semantic_mask = torch.from_numpy(semantic_mask).float().permute(2, 0, 1)
    else:
        semantic_mask = None

    # 如果存储了单帧的点云，用单帧点云投影得到稀疏点云
    if cam_info.pointcloud_camera is not None:
        # 如果可以直接拿到相机视角内的所有点云，就通过相机内参投影到像平面上来计算深度图
        # 但是增量式建图的情况下是没有这个点云的
        h, w = gt_image.shape[1:]
        K = np.eye(3)
        if cam_info.cx:
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
        else:
            K[0, 0] = fov2focal(cam_info.FovX, w)
            K[1, 1] = fov2focal(cam_info.FovY, h)
            K[0, 2] = cam_info.width / 2
            K[1, 2] = cam_info.height / 2
        
        # print(cam_info.image_name)
        # print("center of pcd", cam_info.pointcloud_camera.mean(axis=0)[:3]) # 打印点云中心位置
        pts_depth = np.zeros([1, h, w]) 
        point_camera = cam_info.pointcloud_camera
        uvz = point_camera[point_camera[:, 2] > 0]
        uvz = uvz @ K.T # 投影到像平面
        uvz[:, :2] /= uvz[:, 2:] # 归一化 
        uvz = uvz[uvz[:, 1] >= 0] 
        uvz = uvz[uvz[:, 1] < h]
        uvz = uvz[uvz[:, 0] >= 0]
        uvz = uvz[uvz[:, 0] < w]
        uv = uvz[:, :2]
        uv = uv.astype(int)
        # TODO: may need to consider overlap
        pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]
        # mask = pts_depth>0
        # print("Avg depth of frame: ", id , np.mean(pts_depth[mask]))
        # print("Max depth of frame: ", id , np.max(pts_depth[mask]))
        # print("Min depth of frame: ", id , np.min(pts_depth[mask]))
        gt_pts_depth = torch.from_numpy(pts_depth).float() # 1, H, W
        est_pts_depth = None
  
    # 加载动态MASK
    if cam_info.dynamic_mask is not None:
        if cam_info.dynamic_mask.shape[:2] != resolution[::-1]:
            dynamic_mask = cv2.resize(cam_info.dynamic_mask, resolution)
            # print("对dynamic_mask进行双线性插值")
        else:
            dynamic_mask = cam_info.dynamic_mask
        if len(dynamic_mask.shape) == 2:
            dynamic_mask = dynamic_mask[..., None]
        dynamic_mask = torch.from_numpy(dynamic_mask).float().permute(2, 0, 1) # (1, H, W)
    else:
        dynamic_mask = None

    # 加载ID MASK
    if cam_info.id_mask is not None:
        if cam_info.id_mask.shape[:2] != resolution[::-1]:
            id_mask = cv2.resize(cam_info.id_mask, resolution, interpolation=cv2.INTER_NEAREST)
        else:
            id_mask = cam_info.id_mask
        if len(id_mask.shape) == 2:
            id_mask = id_mask[..., None]
        id_mask = torch.from_numpy(id_mask).int().permute(2, 0, 1) # (1, H, W)
    else:
        id_mask = None

    
    # 加载语义mask
    if cam_info.semantic_mask is not None:
        if cam_info.semantic_mask.shape[:2] != resolution[::-1]:
            semantic_mask = cv2.resize(cam_info.semantic_mask, resolution)
            # print("对semantic_mask进行双线性插值")
        else:
            semantic_mask = cam_info.semantic_mask
        if len(semantic_mask.shape) == 2:
            semantic_mask = semantic_mask[..., None]
        semantic_mask = torch.from_numpy(semantic_mask).float().permute(2, 0, 1)
    else:
        semantic_mask = None

        
    if cam_info.normal_map is not None:
        # 默认没有
        if cam_info.normal_map.shape[:2] != resolution[::-1]:
            normal_map = cv2.resize(cam_info.normal_map, resolution)
        else:
            normal_map = cam_info.normal_map
        normal_map = torch.from_numpy(normal_map).float().permute(2, 0, 1) # H, W, 3-> 3, H, W
    else:
        normal_map = None

    # Camera 对象的初始化
    return Camera(
        colmap_id=cam_info.uid,
        uid=id,
        R=cam_info.R, # np.array(3x3) stored after transpose
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        image=gt_image,
        image_name=cam_info.image_name,
        data_device=args.data_device,
        timestamp=cam_info.timestamp,
        resolution=resolution,
        image_path=cam_info.image_path,
        pts_depth=est_pts_depth,
        gt_pts_depth=gt_pts_depth,
        sky_mask=sky_mask,
        dynamic_mask = dynamic_mask,
        id_mask = id_mask,
        semantic_mask=semantic_mask,
        normal_map = normal_map,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos, bar_format='{l_bar}{bar:50}{r_bar}')):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list




def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    if camera.cx is None:
        camera_entry = {
            "id": id,
            "img_name": camera.image_name,
            "width": camera.width,
            "height": camera.height,
            "position": pos.tolist(),
            "rotation": serializable_array_2d,
            "FoVx": camera.FovX,
            "FoVy": camera.FovY,
        }
    else:
        camera_entry = {
            "id": id,
            "img_name": camera.image_name,
            "width": camera.width,
            "height": camera.height,
            "position": pos.tolist(),
            "rotation": serializable_array_2d,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
        }
    return camera_entry


def calculate_mean_and_std(mean_per_image, std_per_image):
    # Calculate mean RGB across dataset
    mean_dataset = np.mean(mean_per_image, axis=0)

    # Calculate variance of each image
    variances = std_per_image**2

    # Calculate overall variance across the dataset
    overall_variance = np.mean(variances, axis=0) + np.mean((mean_per_image - mean_dataset)**2, axis=0) # (C,)

    # Calculate std RGB across dataset
    std_rgb_dataset = np.sqrt(overall_variance)
    
    return mean_dataset, std_rgb_dataset
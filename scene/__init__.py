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
import random
import json
import torch
from tqdm import tqdm
import numpy as np
from utils.system_utils import searchForMaxIteration
# from scene.gaussian_model import GaussianModel
from scene.dynamic_gaussian_model import DynamicGaussianModel
from scene.gaussian_model import GaussianModel
from scene.envlight import EnvLight
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, calculate_mean_and_std
from scene.waymo_loader import readWaymoInfo
from scene.kittimot_loader import readKittiMotInfo
from scene.emer_waymo_loader import readEmerWaymoInfo
import logging
sceneLoadTypeCallbacks = {
    "Waymo": readWaymoInfo,
    "KittiMot": readKittiMotInfo,
    'EmerWaymo': readEmerWaymoInfo,
}

class Scene:

    gaussians : DynamicGaussianModel # 声明gaussians为GaussianModel类型

    def __init__(self, args, gaussians : DynamicGaussianModel, dynamic_dict:dict, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            logging.info("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        # 1: [Cam1, Cam2, ...],
        # 4: [Cam1, Cam2, ...],
        # 8: [Cam1, Cam2, ...]}
    
        self.test_cameras = {}

        scene_info = sceneLoadTypeCallbacks[args.scene_type](args,dynamic_dict)

        self.dynamic_dict = dynamic_dict
       

        self.time_interval = args.frame_interval # 帧间时间间隔  base.yaml指定为0.02s
        # 从scene_info中更新gs的time_duration而不是args.time_duration
        self.gaussians.time_duration = scene_info.time_duration # 整个场景的持续时间 [-0.48,0.48]48帧

        # 这两个参数管理混乱
        # self.time_interval 在base.yaml中指定为0.02s (但是Waymo数据集的时间间隔是0.1s)
        # scene_info.time_duration 将根据self.time_interval计算，但是子配置文件又额外的指定了一个不生效的time_duration参数【那就不用管】


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            # 从source_path/points3d.ply读取点云数据,是完整的颜色随机的激光点云图，参考wamo_loader.py中 209行
            # 将点云数据复制到 self.model_path 目录下，并命名为 input.ply
            
            # 打包存储相机信息
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.resolution_scales = args.resolution_scales # [1, 2, 4, 8, 16]
        self.scale_index = len(self.resolution_scales) - 1 # 从后往前索引，训练时从粗糙到精细
        for resolution_scale in self.resolution_scales:
            logging.info("Loading Training Cameras at resolution scale {}".format(resolution_scale))
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            logging.info("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            logging.info("Computing nearest_id")


          

            colmap2id = {cam.colmap_id: id for id, cam in enumerate(self.train_cameras[resolution_scale])}
            # 多视角一致性
            with open(os.path.join(self.model_path, "multi_view.json"), 'w') as file:
                json_d = []
                for id, cur_cam in enumerate(tqdm(self.train_cameras[resolution_scale], bar_format='{l_bar}{bar:50}{r_bar}')):

                    # cur_image_name = cur_cam.image_name
                    cur_colmap_id = cur_cam.colmap_id
                    nearest_colmap_id_candidate = [cur_colmap_id - 10, cur_colmap_id + 10, cur_colmap_id - 20, cur_colmap_id + 20]
                    
                    for colmap_id in nearest_colmap_id_candidate:
                        if colmap_id in colmap2id.keys():
                            cur_cam.nearest_id.append(colmap2id[colmap_id])
                            cur_cam.nearest_names.append(self.train_cameras[resolution_scale][colmap2id[colmap_id]].image_name)
                        
                        # near_image_name = "{:03d}_{:1d}".format(colmap_id // 10, colmap_id % 10)
                        # if near_image_name in image_name_to_id_map:
                        #     cur_cam.nearest_id.append(image_name_to_id_map[near_image_name])
                        #     cur_cam.nearest_names.append(near_image_name)


                    # nearest_id应该对应的是self.train_cameras的索引
                    
                    json_d.append({"id": id, 'ref_colmap_id':cur_cam.colmap_id, 'ref_name' : cur_cam.image_name, \
                                   'nearest_id': cur_cam.nearest_id, 'nearest_colmap':[self.train_cameras[resolution_scale][idx].colmap_id for idx in cur_cam.nearest_id], 'nearest_name': [self.train_cameras[resolution_scale][idx].image_name for idx in cur_cam.nearest_id]})
                json.dump(json_d, file)

              

                    
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, 1)
    def upScale(self):
        self.scale_index = max(0, self.scale_index - 1)

    def getTrainCameras(self):
        return self.train_cameras[self.resolution_scales[self.scale_index]]
    
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


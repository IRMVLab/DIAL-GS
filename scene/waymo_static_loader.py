import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud, focal2fov
import logging
from utils.camera_utils import loadCam
import cv2 


ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
LOAD_SIZE = [640, 960]
# LOAD_SIZE = [1280, 1920]  # (H, W) 加载后的图像大小


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius>0:
        scale_factor = 1./fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor




class WaymoDataset:
    def __init__(self, args):
        self.args = args
        self.neg_fov = args.neg_fov # False
        self.cam_num = args.cam_num # 3
        self.resolution_scales = args.resolution_scales # [1, 2, 4, 8, 16]
        self.scale_index = len(self.resolution_scales) - 1 # 从后往前索引，训练时从粗糙到精细
        self.cam_infos = [] # 相机信息列表
        # [
        # [caminfo1, caminfo2, caminfo3], # idx0
        # [caminfo1, caminfo2, caminfo3], # idx1
        #     ...
        # ]


        self.Cameras = {} # 相机字典
        # {
        #     "scale1": [[Cam1,Cam2,Cam3], [Cam1,Cam2,Cam3], ... ,[Cam1,Cam2,Cam3]], 
        #     "scale2": [[Cam1,Cam2,Cam3], [Cam1,Cam2,Cam3], ... ,[Cam1,Cam2,Cam3]], 
        # }

        self.car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(args.source_path, "calib"))) if f.endswith('.txt')] 
        # 只读取部分帧
        # self.car_list = self.car_list[:1]

        self.frame_num = len(self.car_list)
        
        self.cameras_extent = None 
        self.pca_transformation = None
        self.pca_scale_factor = None

        self.pcd = None #融合点云


    def load_cam_info(self):

        # Camera info
        points_all = []
        static_points_all = []
        for idx, car_id in tqdm(enumerate(self.car_list), total=len(self.car_list),desc="Loading Camera Info", bar_format="{l_bar}{bar:50}{r_bar}"):

            ego_pose = np.loadtxt(os.path.join(self.args.source_path, 'pose', car_id + '.txt'))

            with open(os.path.join(self.args.source_path, 'calib', car_id + '.txt')) as f:
                calib_data = f.readlines()
                L = [list(map(float, line.split()[1:])) for line in calib_data] # 长度为行数，每行12个float元素

            Ks = np.array(L[:5]).reshape(-1, 3, 4)[:, :, :3] # 相机内参旋转矩阵
            lidar2cam = np.array(L[-5:]).reshape(-1, 3, 4)
            lidar2cam = pad_poses(lidar2cam)

            cam2lidar = np.linalg.inv(lidar2cam)
            c2w = ego_pose @ cam2lidar # world_T_lidar @ lidar_T_cam
            w2c = np.linalg.inv(c2w) # (N_cam, 4, 4)  # world_T_cam
            

            images = []
            image_paths = []
            HWs = []
            for subdir in ['image_0', 'image_1', 'image_2', 'image_3', 'image_4'][:self.args.cam_num]:
                image_path = os.path.join(self.args.source_path, subdir, car_id + '.png')
                im_data = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR uint8
                im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)  # 转成 RGB
                im_data = cv2.resize(im_data, (LOAD_SIZE[1], LOAD_SIZE[0]), interpolation=cv2.INTER_LINEAR)  #(W, H)作为参数 以线性插值法缩放图像
                H, W = im_data.shape[:2]
                image = im_data.astype(np.float32) / 255.0  # im_data shape: (H, W, 3) float32
                # im_data = Image.open(image_path)
                # im_data = im_data.resize((LOAD_SIZE[1], LOAD_SIZE[0]), Image.BILINEAR) # PIL resize: (W, H) 以双线性插值法缩放图像
                # W, H = im_data.size
                # image = np.array(im_data) / 255. # 归一化到[0,1] 
                HWs.append((H, W)) # (H, W)是图像加载后的大小，不是原始大小
                images.append(image)
                image_paths.append(image_path)

            sky_masks = []
            for subdir in ['sky_0', 'sky_1', 'sky_2', 'sky_3', 'sky_4'][:self.args.cam_num]:
                sky_path = os.path.join(self.args.source_path, subdir, car_id + '.png')
                sky_data = cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)
                sky_data = cv2.resize(sky_data, (LOAD_SIZE[1], LOAD_SIZE[0]), interpolation=cv2.INTER_NEAREST)
                sky_mask = (sky_data > 0).astype(np.float32) # 1 表示天空，0表示非天空
                sky_masks.append(sky_mask)

            semantic_masks = []

            for subdir in ['semantic_mask_0','semantic_mask_1','semantic_mask_2','semantic_mask_3','semantic_mask_4'][:self.args.cam_num]:
                semantic_path = os.path.join(self.args.source_path, subdir, car_id + '.png')
                semantic_data = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
                semantic_data = cv2.resize(semantic_data, (LOAD_SIZE[1], LOAD_SIZE[0]), interpolation=cv2.INTER_NEAREST)
                semantic_mask = (semantic_data > 0).astype(np.float32)  # 1表示人车
                # 膨胀semantic_mask
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 30))  # 横宽窄，纵高高
                dilated = cv2.dilate(semantic_mask, kernel, iterations=1)
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
                closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
                semantic_masks.append(closed)
            
            
            point = np.fromfile(os.path.join(self.args.source_path, "velodyne", car_id + ".bin"),
                            dtype=np.float32, count=-1).reshape(-1, 6)
 
            point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
            point_xyz_world = (np.pad(point_xyz, (0, 1), constant_values=1) @ ego_pose.T)[:, :3] # (0,1)表示在点云的最后一列添加，转换到世界坐标系后所有的z坐标都是负数
            points_all.append(point_xyz_world)

            self.cam_infos.append([])
            frame_static_points = []
            frame_static_points_num = 0
            for j in range(self.args.cam_num):
                point_camera = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ lidar2cam[j].T)[:, :3]  # 相机坐标系下的点云
                # 将点云从世界坐标系转换到第j个相机坐标系
                R = np.transpose(w2c[j, :3, :3])  # R是转置存储的w2c旋转矩阵 !!!
                T = w2c[j, :3, 3]
                K = Ks[j]
               
                fx = float(K[0, 0]) * LOAD_SIZE[1] / ORIGINAL_SIZE[j][1]
                fy = float(K[1, 1]) * LOAD_SIZE[0] / ORIGINAL_SIZE[j][0]
                cx = float(K[0, 2]) * LOAD_SIZE[1] / ORIGINAL_SIZE[j][1]
                cy = float(K[1, 2]) * LOAD_SIZE[0] / ORIGINAL_SIZE[j][0]
                
                width=HWs[j][1]
                height=HWs[j][0]
                if self.neg_fov:
                    FovY = -1.0
                    FovX = -1.0
                else:
                    FovY = focal2fov(fy, height)
                    FovX = focal2fov(fx, width)


                # 获取语义mask外的点云
                valid_points = point_camera[:, 2] > 0.05  # z值需要大于0（在相机前方） (N, 3)
                if np.sum(valid_points) > 0:
                    points_valid = point_camera[valid_points]
                    
                    # 投影到像素坐标
                    pixels_x = (points_valid[:, 0] / points_valid[:, 2] * fx + cx).astype(np.int32)
                    pixels_y = (points_valid[:, 1] / points_valid[:, 2] * fy + cy).astype(np.int32)
                    
                    # 过滤出在图像范围内的点
                    valid_pixels = (pixels_x >= 0) & (pixels_x < width) & (pixels_y >= 0) & (pixels_y < height)
                    if np.sum(valid_pixels) > 0:
                        pixels_x = pixels_x[valid_pixels]
                        pixels_y = pixels_y[valid_pixels]
                        
                        # 获取这些点在dynamic_mask中的值
                        mask_values = semantic_masks[j][pixels_y, pixels_x]
                        
                        # 找出落在非动态区域的点(mask值为0)
                        static_mask = (mask_values == 0)
                        if np.sum(static_mask) > 0:
                            # 获取原始点云索引
                            static_indices = np.where(valid_points)[0][valid_pixels][static_mask]
                            # 添加到静态点云列表
                            frame_static_points.append(point_xyz_world[static_indices])
                            frame_static_points_num += len(static_indices)
                
                
                # cam_infos内容
                self.cam_infos[idx].append(CameraInfo(uid=idx * self.cam_num + j, R=R, T=T, FovY=FovY, FovX=FovX,
                                            image=images[j], 
                                            image_path=image_paths[j], image_name=car_id,
                                            width=HWs[j][1], height=HWs[j][0], 
                                            pointcloud_camera = point_camera,
                                            fx=fx, fy=fy, cx=cx, cy=cy, 
                                            sky_mask=sky_masks[j], 
                                            semantic_mask=semantic_masks[j],
                                            normal_map=None))
                
            if len(frame_static_points) > 0:
                frame_static_points = np.concatenate(frame_static_points, axis=0)
                static_points_all.append(frame_static_points)

        
        points_all = np.concatenate(points_all, axis=0) # (N, 3)
        indices = np.random.choice(points_all.shape[0], self.args.refine_init_pcd_limit, replace=True) # 采样，限制初始化点云数
        points_all = points_all[indices]
        self.pcd = points_all

        # 处理静态点云
        if len(static_points_all) > 0:
            static_points_all = np.concatenate(static_points_all, axis=0)  # (N, 3)
            # 如果静态点云数量太多，进行下采样
            if static_points_all.shape[0] > self.args.refine_init_pcd_limit:
                indices = np.random.choice(static_points_all.shape[0], self.args.refine_init_pcd_limit, replace=False)
                static_points_all = static_points_all[indices]
            self.static_pcd = static_points_all
        else:
            print("警告：没有找到静态点云")
            self.static_pcd = np.zeros((0, 3))
        
        # PCA 变换
        w2cs = np.zeros((self.frame_num * self.cam_num, 4, 4))
        Rs = []
        Ts = []
        for frame_id in range(self.frame_num):
            for cam_id in range(self.cam_num):
                Rs.append(self.cam_infos[frame_id][cam_id].R)
                Ts.append(self.cam_infos[frame_id][cam_id].T)
        Rs = np.stack(Rs, axis=0) # (frame_num * cam_num, 3, 3)
        Ts = np.stack(Ts, axis=0) # (frame_num * cam_num, 3)
        
        w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
        w2cs[:, :3, 3] = Ts
        w2cs[:, 3, 3] = 1
        c2ws = unpad_poses(np.linalg.inv(w2cs))
        c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=self.args.fix_radius) # 进行PCA变换
        self.pca_transformation = transform
        self.pca_scale_factor = scale_factor
        c2ws = pad_poses(c2ws)
        # print("PCA transformation matrix:\n", transform)
        # print("PCA scale factor:", scale_factor)

        for frame_id in range(self.frame_num):
            for cam_id in range(self.cam_num):
                idx = frame_id * self.cam_num + cam_id
                cam_info = self.cam_infos[frame_id][cam_id]
                c2w = c2ws[idx]
                w2c = np.linalg.inv(c2w)
                cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                cam_info.T[:] = w2c[:3, 3]
                cam_info.pointcloud_camera[:] *= scale_factor
        if self.pca_transformation is not None:
            self.pcd = (np.pad(self.pcd, ((0, 0), (0, 1)), constant_values=1) @ self.pca_transformation.T)[:, :3] # 转换到PCA坐标系
            if self.static_pcd.shape[0] > 0:
                self.static_pcd = (np.pad(self.static_pcd, ((0, 0), (0, 1)), constant_values=1) @ self.pca_transformation.T)[:, :3]
        


    def load_Cam(self):
        for resolution_scale in self.resolution_scales:
            logging.info("Loading Training Cameras at resolution scale {}".format(resolution_scale))
            self.Cameras[f"scale{resolution_scale}"] = []
            for idx in tqdm(range(self.frame_num), desc=f"Loading data at different scales"):
                self.Cameras[f"scale{resolution_scale}"].append([])
                for cam_j in range(self.cam_num):
                    cam_info = self.cam_infos[idx][cam_j]
                    uid = f"id{idx}_scale{resolution_scale}_cam{cam_j}"
                    self.Cameras["scale"+str(resolution_scale)][idx].append(loadCam(self.args, uid, cam_info, resolution_scale))


    def load(self):
        self.load_cam_info()   # 加载相机信息
        flattened_cam_infos = [cam for sublist in self.cam_infos for cam in sublist] # 将相机信息列表展平
        nerf_normalization = getNerfppNorm(flattened_cam_infos)
        nerf_normalization['radius'] = 1/nerf_normalization['radius']
        self.cameras_extent = nerf_normalization['radius'] # 计算相机的包围盒大小
        self.load_Cam()
        
    def getTrainCameras(self):
        Cams = self.Cameras["scale1"]
        flattened_cams = [cam for sublist in Cams for cam in sublist]
        return flattened_cams
    
    def getTestCameras(self):
        return None

    def __len__(self):
        return self.frame_num * self.cam_num 

    def get_cam(self, scale, frame_idx=None, cam_idx=None):
        """
        获取 WaymoDataset 中的相机数据。
        
        参数:
            scale (int): 分辨率缩放比例，必须在 self.resolution_scales 中。
            frame_idx (int, 可选): 帧索引，必须在 [0, self.frame_num) 范围内。
            cam_idx (int, 可选): 相机索引，必须在 [0, self.cam_num) 范围内。
        
        返回:
            根据传入参数返回对应的相机数据:
            - 如果只传入 scale: 返回 self.Cameras["scale*"]。
            - 如果传入 scale 和 frame_idx: 返回 self.Cameras["scale1"][frame_idx]。
            - 如果传入 scale, frame_idx 和 cam_idx: 返回 self.Cameras["scale*"][frame_idx][cam_idx]。
        
        抛出:
            IndexError: 如果 frame_idx 或 cam_idx 超出范围。
            ValueError: 如果 scale 不在 self.resolution_scales 中。
        """
        # 检查 scale 是否有效
        if scale not in self.resolution_scales:
            raise ValueError(f"Invalid scale value '{scale}'. Must be one of the following: {self.resolution_scales}")

        # 如果只传入 scale
        if frame_idx is None and cam_idx is None:
            return self.Cameras[f"scale{scale}"]

        # 检查 frame_idx 是否有效
        if frame_idx is not None and (frame_idx < 0 or frame_idx >= self.frame_num):
            raise IndexError(f"Invalid frame_idx value '{frame_idx}'. Must be in range [0, {self.frame_num}).")

        # 如果只传入 scale 和 frame_idx
        if cam_idx is None:
            return self.Cameras[f"scale{scale}"][frame_idx]

        # 检查 cam_idx 是否有效
        if cam_idx is not None and (cam_idx < 0 or cam_idx >= self.cam_num):
            raise IndexError(f"Invalid cam_idx value '{cam_idx}'. Must be in range [0, {self.cam_num}).")

        # 如果传入 scale, frame_idx 和 cam_idx
        return self.Cameras[f"scale{scale}"][frame_idx][cam_idx]




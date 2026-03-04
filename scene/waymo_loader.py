import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud, focal2fov
import cv2


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


def readWaymoInfo(args,dynamic_dict):
    neg_fov = args.neg_fov 
    cam_infos = []
    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(args.source_path, "calib"))) if f.endswith('.txt')]
    # 只读取部分帧
    # car_list = car_list[:3]
    points = []
    points_time = []
    
    load_size = [640, 960]
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    frame_num = len(car_list)
    if args.frame_interval > 0:
        time_duration = [-args.frame_interval*(frame_num-1)/2,args.frame_interval*(frame_num-1)/2]
        # -0.02*(49-1)/2 = -0.48
    else:
        time_duration = args.time_duration
        # base.yaml指定了frame_interval，所以该else语句不会被执行

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        ego_pose = np.loadtxt(os.path.join(args.source_path, 'pose', car_id + '.txt')) # (4, 4) 
        # ego_pose => lidar2world

        # CAMERA DIRECTION: RIGHT DOWN FORWARDS
        with open(os.path.join(args.source_path, 'calib', car_id + '.txt')) as f:
            calib_data = f.readlines()
            L = [list(map(float, line.split()[1:])) for line in calib_data]

        Ks = np.array(L[:5]).reshape(-1, 3, 4)[:, :, :3] # 相机内参矩阵
        lidar2cam = np.array(L[-5:]).reshape(-1, 3, 4) # (5, 3, 4)
        lidar2cam = pad_poses(lidar2cam) # (5, 4, 4) 

        cam2lidar = np.linalg.inv(lidar2cam) 
        c2w = ego_pose @ cam2lidar    # lidar2world @ cam2lidar = cam2world
        w2c = np.linalg.inv(c2w)
        images = []
        image_paths = []
        HWs = []
        for subdir in ['image_0', 'image_1', 'image_2', 'image_3', 'image_4'][:args.cam_num]:
            image_path = os.path.join(args.source_path, subdir, car_id + '.png')
            im_data = Image.open(image_path)
            im_data = im_data.resize((load_size[1], load_size[0]), Image.BILINEAR) # PIL resize: (W, H) 以双线性插值法缩放图像
            W, H = im_data.size
            image = np.array(im_data) / 255. # 归一化到[0,1] 
            HWs.append((H, W)) # (H, W)是图像加载后的大小，不是原始大小
            images.append(image)
            image_paths.append(image_path)

        sky_masks = []
        for subdir in ['sky_0', 'sky_1', 'sky_2', 'sky_3', 'sky_4'][:args.cam_num]:
            sky_data = Image.open(os.path.join(args.source_path, subdir, car_id + '.png'))
            sky_data = sky_data.resize((load_size[1], load_size[0]), Image.NEAREST) # PIL resize: (W, H)
            sky_mask = np.array(sky_data)>0 # sky_mask是一个布尔数组，True表示天空区域
            sky_masks.append(sky_mask.astype(np.float32)) # 将布尔数组转换为float32类型的数组，True变为1.0，False变为0.0
        
        normal_maps = []
        if args.load_normal_map: 
            for subdir in ['normal_0', 'normal_1', 'normal_2', 'normal_3', 'normal_4'][:args.cam_num]:
                normal_data = Image.open(os.path.join(args.source_path, subdir, car_id + '.png'))
                normal_data = normal_data.resize((load_size[1], load_size[0]), Image.BILINEAR)
                normal_map = (np.array(normal_data)) / 255.
                normal_maps.append(normal_map)  

        dynamic_masks = []
        id_masks = []

        dilate_size = 2 # 可以调整膨胀的像素数量
        kernel = np.zeros((dilate_size + 1, 1), dtype=np.uint8)
        kernel[:, 0] = 1  # 竖直结构元素，只向下膨胀
        def dilate_downward(mask):
            """使用OpenCV实现只向下的膨胀，anchor固定在核的左上角"""
            return cv2.dilate(mask, kernel, iterations=1, anchor=(0, 0))
        
        dynamic_ids = sorted(dynamic_dict["cam_0"])
        id_mapping = {v: i for i, v in enumerate(dynamic_ids)}
        print(f"ID mapping for cam_0: {id_mapping}")
        if args.load_dynamic_mask:
            for subdir in ['tracking_0', 'tracking_1', 'tracking_2', 'tracking_3', 'tracking_4'][:args.cam_num]:
                cam_name = subdir.split('_')[-1]
                dynamic_ids = dynamic_dict["cam_" + cam_name] # dynamic_dict来自configs/waymo_stage_2.sh中的dynmaic_id_dict_path参数
                # print(f"Dynamic ids for {cam_name}: {dynamic_ids}")
                tracking_path = os.path.join(args.source_path, subdir, car_id + '.npy')
                tracking_data = np.load(tracking_path)
                tracking_data = cv2.resize(tracking_data, (load_size[1], load_size[0]), interpolation=cv2.INTER_NEAREST)
                dynamic_mask = np.zeros_like(tracking_data, dtype=np.float32)
                id_mask = np.ones_like(tracking_data, dtype=np.int32) * (-1) # -1表示静态部分
                for dynamic_id in dynamic_ids:
                    dynamic_mask[tracking_data == dynamic_id] = 1.0
                # 只对0号相机进行id监督
                    if  subdir == 'tracking_0': 
                        id_mask[tracking_data == dynamic_id] = id_mapping[dynamic_id] # 0,1,2,...表示不同的动态物体
                        # 对id_mask进行竖直向下膨胀（需要特殊处理，因为包含多个ID值）
                        id_mask_dilated = np.full_like(id_mask, -1)
                        for valid_id in id_mapping.values():
                            # 创建当前ID的二值mask
                            current_id_mask = (id_mask == valid_id).astype(np.uint8)
                            
                            # 对当前ID mask进行竖直向下膨胀
                            current_id_dilated = dilate_downward(current_id_mask)
                            
                            # 将膨胀后的区域分配给对应的新ID
                            id_mask_dilated[current_id_dilated > 0] = valid_id
                        
                        id_mask = id_mask_dilated


                dynamic_masks.append(dynamic_mask)
                id_masks.append(id_mask)

        

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (len(car_list) - 1) 
        # -0.48 + (0.48 - (-0.48)) * id / (49 - 1)  真实时间尺度放缩到index尺度后再计算真实时间步
        point = np.fromfile(os.path.join(args.source_path, "velodyne", car_id + ".bin"),
                            dtype=np.float32, count=-1).reshape(-1, 6) # count=-1表示读取所有数据
        point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1) # 这时候的z有正有负
        # point_xyz = np.zeros((5, 3), dtype=np.float32)
        # print(f"Original point num of frame{idx} ", point.shape[0])
        # point[:, :3]：前三列 [x, y, z]，即点的三维坐标。
        # point[:, 3:4]：第四列 intensity，即点的强度。
        # point[:, 4:5]：第五列 elongation，即点的拉伸值。
        # point[:, 5:]：第六列 timestamp_pts，即点的时间戳。
       
        # lidar2world @ point[:, :3]  ==> point in world coordinate
        point_xyz_world = (np.pad(point_xyz, (0, 1), constant_values=1) @ ego_pose.T)[:, :3] # (0,1)表示在点云的最后一列添加，转换到世界坐标系后所有的z坐标都是负数
        points.append(point_xyz_world)
        point_time = np.full_like(point_xyz_world[:, :1], timestamp) # (N,1)的数组，填充为当前时间戳
        points_time.append(point_time)
        for j in range(args.cam_num):
            point_camera = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ lidar2cam[j].T)[:, :3] 
            # 将点云从世界坐标系转换到第j个相机坐标系
            R = np.transpose(w2c[j, :3, :3])  # shape (3, 3)
            T = w2c[j, :3, 3]
            K = Ks[j]
            fx = float(K[0, 0]) * load_size[1] / ORIGINAL_SIZE[j][1]
            fy = float(K[1, 1]) * load_size[0] / ORIGINAL_SIZE[j][0]
            cx = float(K[0, 2]) * load_size[1] / ORIGINAL_SIZE[j][1]
            cy = float(K[1, 2]) * load_size[0] / ORIGINAL_SIZE[j][0]
            width=HWs[j][1]
            height=HWs[j][0]
            if neg_fov:
                FovY = -1.0
                FovX = -1.0
            else:
                FovY = focal2fov(fy, height)
                FovX = focal2fov(fx, width)
            cam_infos.append(CameraInfo(uid=idx * 5 + j, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image=images[j], 
                                        image_path=image_paths[j], image_name=car_id,
                                        width=HWs[j][1], height=HWs[j][0], timestamp=timestamp,
                                        pointcloud_camera = point_camera,
                                        fx=fx, fy=fy, cx=cx, cy=cy, 
                                        sky_mask=sky_masks[j], 
                                        dynamic_mask=dynamic_masks[j] if args.load_dynamic_mask else None,
                                        id_mask=id_masks[j] if args.load_dynamic_mask else None,
                                        normal_map=normal_maps[j] if args.load_normal_map else None))

        if args.debug_cuda:
            break

    pointcloud = np.concatenate(points, axis=0) # (N, 3) 世界坐标系 #z方向是负数
    print(f"Point num of all frames: {pointcloud.shape[0]}")


    pointcloud_timestamp = np.concatenate(points_time, axis=0) # (N, 1)
    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True) #第一个参数是点云的数量，第二个参数是要采样的点数，replace=True表示可以重复采样
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]
    
    # ================== PCA transform ==================
    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius) # 进行PCA变换
    print(f"PCA transform: {transform}")
    print(f"PCA scale_factor: {scale_factor}")

    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3] #PCA纠正后的世界坐标系点云
    if args.eval: 
        # 默认False
        # ## for snerf scene
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold == 0]

        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]
        
        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num)>0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1/nerf_normalization['radius']

    ply_path = os.path.join(args.source_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # 保存一个随机颜色的点云
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    
    # 源代码冗余
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0],3]), normals=None, time=pointcloud_timestamp) #颜色全为0
    time_interval = (time_duration[1] - time_duration[0]) / (len(car_list) - 1) # 如果args.time_interval>0,这里算出来的还是args.time_interval，否则将会根据给定的timeduration计算interval

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval,
                           time_duration=time_duration)

    return scene_info

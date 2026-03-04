# Description: Load the EmerWaymo dataset for training and testing
# adapted from the PVG datareader for the data from EmerNeRF

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud, focal2fov

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


def readEmerWaymoInfo(args):
    
    eval = args.eval
    load_sky_mask = args.load_sky_mask
    load_panoptic_mask = args.load_panoptic_mask
    load_sam_mask = args.load_sam_mask
    load_dynamic_mask = args.load_dynamic_mask
    load_normal_map = args.load_normal_map
    load_feat_map = args.load_feat_map
    load_intrinsic = args.load_intrinsic
    load_c2w = args.load_c2w
    save_occ_grid = args.save_occ_grid
    occ_voxel_size = args.occ_voxel_size
    recompute_occ_grid = args.recompute_occ_grid
    use_bg_gs = args.use_bg_gs
    white_background = args.white_background
    neg_fov = args.neg_fov
    

    num_pts = args.num_pts
    start_time = args.start_time
    end_time = args.end_time
    stride = args.stride
    original_start_time = args.original_start_time     

    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    load_size = [640, 960]

    cam_infos = []
    points = []
    points_time = []

    data_root = args.source_path
    image_folder = os.path.join(data_root, "images")
    num_seqs = len(os.listdir(image_folder))/5
    if end_time == -1:
        end_time = int(num_seqs)
    else:
        end_time += 1

    frame_num = end_time - start_time
    # assert frame_num == 50, "frame_num should be 50"
    time_duration = args.time_duration
    time_interval = (time_duration[1] - time_duration[0]) / (end_time - start_time)
    
    camera_list = [0, 1, 2]
    truncated_min_range, truncated_max_range = -2, 80
    
    
    # ---------------------------------------------
    # load poses: intrinsic, c2w, l2w per camera
    # ---------------------------------------------
    _intrinsics = []
    cam_to_egos = []
    for i in camera_list:
        # load intrinsics
        intrinsic = np.loadtxt(os.path.join(data_root, "intrinsics", f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * load_size[1] / ORIGINAL_SIZE[i][1],
            fy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        cx, cy = (
            cx * load_size[1] / ORIGINAL_SIZE[i][1],
            cy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _intrinsics.append(intrinsic)
        # load extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_root, "extrinsics", f"{i}.txt"))
        # opencv coordinate system: x right, y down, z front
        # waymo coordinate system: x front, y left, z up
        cam_to_egos.append(cam_to_ego @ OPENCV2DATASET) # opencv_cam -> waymo_cam -> waymo_ego


    # ---------------------------------------------
    # get c2w and w2c transformation per frame and camera 
    # ---------------------------------------------
    # compute per-image poses and intrinsics
    cam_to_worlds, ego_to_worlds = [], []
    intrinsics, cam_ids = [], []
    lidar_to_worlds = []
    # ===! for waymo, we simplify timestamps as the time indices
    timestamps, timesteps = [], []
    # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
    # the first ego pose as the origin of the world coordinate system.
    ego_to_world_start = np.loadtxt(os.path.join(data_root, "ego_pose", f"{start_time:03d}.txt"))
    for t in range(start_time, end_time):
        ego_to_world_current = np.loadtxt(os.path.join(data_root, "ego_pose", f"{t:03d}.txt"))
        # ego to world transformation: cur_ego -> world -> start_ego(world)
        ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
        ego_to_worlds.append(ego_to_world)
        for cam_id in camera_list:
            cam_ids.append(cam_id)
            # transformation:
            # opencv_cam -> waymo_cam -> waymo_cur_ego -> world -> start_ego(world)
            cam2world = ego_to_world @ cam_to_egos[cam_id]
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics[cam_id])
            # ===! we use time indices as the timestamp for waymo dataset for simplicity
            # ===! we can use the actual timestamps if needed
            # to be improved
            timestamps.append(t - start_time)
            timesteps.append(t - start_time)
        # lidar to world : lidar = ego in waymo
        lidar_to_worlds.append(ego_to_world)
    # convert to numpy arrays
    intrinsics = np.stack(intrinsics, axis=0)
    cam_to_worlds = np.stack(cam_to_worlds, axis=0)
    ego_to_worlds = np.stack(ego_to_worlds, axis=0)
    lidar_to_worlds = np.stack(lidar_to_worlds, axis=0)
    cam_ids = np.array(cam_ids)
    timestamps = np.array(timestamps)
    timesteps = np.array(timesteps)


    # ---------------------------------------------
    # get image, sky_mask, lidar per frame and camera 
    # --------------------------------------------- 
    accumulated_num_original_rays = 0
    accumulated_num_rays = 0    
    
    for idx, t in enumerate(tqdm(range(start_time, end_time), desc="Loading data", bar_format='{l_bar}{bar:50}{r_bar}')):
  
        images = []
        image_paths = []
        HWs = []
        sky_masks = []
        dynamic_masks = []
        normal_maps = []
        
        for cam_idx in camera_list:
            image_path = os.path.join(args.source_path, "images", f"{t:03d}_{cam_idx}.jpg")
            im_data = Image.open(image_path)
            im_data = im_data.resize((load_size[1], load_size[0]), Image.BILINEAR) # PIL resize: (W, H)
            W, H = im_data.size
            image = np.array(im_data) / 255.
            HWs.append((H, W))
            images.append(image)
            image_paths.append(image_path)

            sky_path = os.path.join(args.source_path, "sky_masks", f"{t:03d}_{cam_idx}.png")
            sky_data = Image.open(sky_path)
            sky_data = sky_data.resize((load_size[1], load_size[0]), Image.NEAREST) # PIL resize: (W, H)
            sky_mask = np.array(sky_data)>0
            sky_masks.append(sky_mask.astype(np.float32))
            
            if load_normal_map:
                normal_path = os.path.join(args.source_path, "normals", f"{t:03d}_{cam_idx}.jpg")
                normal_data = Image.open(normal_path)
                normal_data = normal_data.resize((load_size[1], load_size[0]), Image.BILINEAR)
                normal_map = (np.array(normal_data)) / 255. # [0, 1]
                normal_maps.append(normal_map)            
            
            if load_dynamic_mask:
                dynamic_path = os.path.join(args.source_path, "dynamic_masks", f"{t:03d}_{cam_idx}.png")
                dynamic_data = Image.open(dynamic_path)
                dynamic_data = dynamic_data.resize((load_size[1], load_size[0]), Image.BILINEAR)
                dynamic_mask = np.array(dynamic_data)>0
                dynamic_masks.append(dynamic_mask.astype(np.float32))
            

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (frame_num - 1)
        lidar_info = np.memmap(
                os.path.join(data_root, "lidar", f"{t:03d}.bin"),
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 10) 
        original_length = len(lidar_info)
        accumulated_num_original_rays += original_length
        lidar_origins = lidar_info[:, :3]
        lidar_points = lidar_info[:, 3:6]
        lidar_ids = lidar_info[:, -1]
        # select lidar points based on a truncated ego-forward-directional range
        # make sure most of lidar points are within the range of the camera
        valid_mask = lidar_points[:, 0] < truncated_max_range
        valid_mask = valid_mask & (lidar_points[:, 0] > truncated_min_range)
        lidar_origins = lidar_origins[valid_mask]
        lidar_points = lidar_points[valid_mask]
        lidar_ids = lidar_ids[valid_mask]
        # transform lidar points to world coordinate system
        lidar_origins = (
            lidar_to_worlds[idx][:3, :3] @ lidar_origins.T
            + lidar_to_worlds[idx][:3, 3:4]
        ).T
        lidar_points = (
            lidar_to_worlds[idx][:3, :3] @ lidar_points.T
            + lidar_to_worlds[idx][:3, 3:4]
        ).T # point_xyz_world
        
        points.append(lidar_points)
        point_time = np.full_like(lidar_points[:, :1], timestamp)
        points_time.append(point_time)
        
        for cam_idx in camera_list:
            # world-lidar-pts --> camera-pts : w2c
            c2w = cam_to_worlds[int(len(camera_list))*idx + cam_idx]
            w2c = np.linalg.inv(c2w)
            point_camera = (
                w2c[:3, :3] @ lidar_points.T
                + w2c[:3, 3:4]
            ).T

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            K = _intrinsics[cam_idx]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            height, width = HWs[cam_idx]
            if neg_fov:
                FovY = -1.0
                FovX = -1.0
            else:
                FovY = focal2fov(fy, height)
                FovX = focal2fov(fx, width)
            cam_infos.append(CameraInfo(uid=idx * 10 + cam_idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image=images[cam_idx], 
                                        image_path=image_paths[cam_idx], image_name=f"{t:03d}_{cam_idx}",
                                        width=width, height=height, timestamp=timestamp,
                                        pointcloud_camera = point_camera,
                                        fx=fx, fy=fy, cx=cx, cy=cy, 
                                        sky_mask=sky_masks[cam_idx],
                                        dynamic_mask=dynamic_masks[cam_idx] if load_dynamic_mask else None,
                                        normal_map=normal_maps[cam_idx] if load_normal_map else None,))

        if args.debug_cuda:
            break

    pointcloud = np.concatenate(points, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)

    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data", bar_format='{l_bar}{bar:50}{r_bar}')):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    if args.eval:
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
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0],3]), normals=None, time=pointcloud_timestamp)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval,
                           time_duration=time_duration)

    return scene_info

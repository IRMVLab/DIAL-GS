import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud, getWorld2View2, focal2fov, fov2focal
from utils.feature_extractor import extract_and_save_features
from tqdm import trange
from utils.sh_utils import SH2RGB
from utils.general_utils import sample_on_aabb_surface, get_OccGrid, GridSample3D
from utils.image_utils import get_robust_pca, get_panoptic_id
from pathlib import Path

def constructCameras_waymo(frames_list, white_background, mapper = {},
                           load_intrinsic=False, load_c2w=False, start_time = 50, original_start_time = 0):
    cam_infos = []
    for idx, frame in enumerate(frames_list):
        # current frame time
        time = mapper[frame["time"]]
        # ------------------
        # load c2w
        # ------------------
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to OpenCV/COLMAP (Y down, Z forward)
        #c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        # ------------------
        # load image
        # ------------------
        cam_name = image_path = frame['file_path']
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        load_size = frame["load_size"]
        #image = PILtoTorch(image, load_size) #(800,800))
        # resize to load_size
        image = image.resize(load_size, Image.BILINEAR)
        image = np.array(image) / 255.0
        
        # ------------------
        # load depth-map
        # ------------------
        depth_map = frame.get('depth_map', None)

        
        # ------------------
        # load sky-mask
        # ------------------
        sky_mask_path, sky_mask = frame["sky_mask_path"], None
        if sky_mask_path is not None:
            sky_mask = Image.open(sky_mask_path)
            sky_mask = sky_mask.resize(load_size, Image.BILINEAR)
            sky_mask = np.array(sky_mask) > 0
            sky_mask = sky_mask.astype(np.float32)
        # ------------------
        # load intrinsic
        # ------------------
        # intrinsic to fov: intrinsic 已经被 scale
        intrinsic = frame["intrinsic"]
        fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
        # get fov
        fovx = focal2fov(fx, image.shape[1])
        fovy = focal2fov(fy, image.shape[0])
        FovY = fovy
        FovX = fovx

        # ------------------
        # load semantic mask
        # ------------------
        semantic_mask_path, semantic_mask = frame["semantic_mask_path"], None
        if semantic_mask_path is not None:
            semantic_mask = np.load(semantic_mask_path)
            semantic_mask = Image.fromarray(semantic_mask.squeeze(-1))
            semantic_mask = semantic_mask.resize(load_size, Image.NEAREST)
            # to numpy
            #semantic_mask = np.array(semantic_mask)#  .unsqueeze(-1)

        # ------------------
        # load instance mask
        # ------------------
        instance_mask_path, instance_mask = frame["instance_mask_path"], None
        if instance_mask_path is not None:
            instance_mask = np.load(instance_mask_path)
            instance_mask = Image.fromarray(instance_mask.squeeze(-1))
            instance_mask = instance_mask.resize(load_size, Image.NEAREST)
            # to numpy
            #instance_mask = np.array(instance_mask) #.unsqueeze(-1)

        # ------------------
        # load sam mask
        # ------------------
        sam_mask_path, sam_mask = frame["sam_mask_path"], None
        if sam_mask_path is not None:
            sam_mask = Image.open(sam_mask_path)
            sam_mask = sam_mask.resize(load_size, Image.NEAREST)
            # to numpy
            #sam_mask = np.array(sam_mask) #.unsqueeze(-1)

        # ------------------
        # load dynamic mask
        # ------------------
        dynamic_mask_path, dynamic_mask = frame["dynamic_mask_path"], None
        if dynamic_mask_path is not None:
            dynamic_mask = Image.open(dynamic_mask_path)
            dynamic_mask = dynamic_mask.resize(load_size, Image.NEAREST)
            # to numpy
            dynamic_mask = np.array(dynamic_mask) 

        # ------------------
        # load feat map
        # ------------------
        feat_map_path, feat_map = frame["feat_map_path"], None
        if feat_map_path is not None:
            # mmap_mode="r" is to avoid memory overflow when loading features
            # but it only slightly helps... do we have a better way to load features?
            features = np.load(feat_map_path, mmap_mode="r").squeeze()
            # Create a writable copy of the array
            features = np.array(features, copy=True)
            features = torch.from_numpy(features).unsqueeze(0).float()

            # shape: (num_imgs, num_patches_h, num_patches_w, C)
            # featmap_downscale_factor is used to convert the image coordinates to ViT feature coordinates.
            # resizing ViT features to (H, W) using bilinear interpolation is infeasible.
            # imagine a feature array of shape (num_timesteps x num_cams, 640, 960, 768). it's too large to fit in GPU memory.
            featmap_downscale_factor = (
                features.shape[1] / 640,
                features.shape[2] / 960,
            )
            # print(
            #     f"Loaded {features.shape} dinov2_vitb14 features."
            # )
            # print(f"Feature scale: {featmap_downscale_factor}")
            # print(f"Computing features PCA...")
            # compute feature visualization matrix
            C = features.shape[-1]
            # no need to compute PCA on the entire set of features, we randomly sample 100k features
            temp_feats = features.reshape(-1, C)
            max_elements_to_compute_pca = min(100000, temp_feats.shape[0])
            selected_features = temp_feats[
                np.random.choice(
                    temp_feats.shape[0], max_elements_to_compute_pca, replace=False
                )
            ]
            target_feature_dim = 3
            device='cuda'
            if target_feature_dim is not None:
                # print(
                #     f"Reducing features to {target_feature_dim} dimensions."
                # )
                # compute PCA to reduce the feature dimension to target_feature_dim
                U, S, reduce_to_target_dim_mat = torch.pca_lowrank(
                    selected_features, q=target_feature_dim, niter=20
                )
                # compute the fraction of variance explained by target_feature_dim
                variances = S**2
                fraction_var_explained = variances / variances.sum()
                # print(f"[PCA] fraction_var_explained: \n{fraction_var_explained}")
                # print(
                #     f"[PCA] fraction_var_explained sum: {fraction_var_explained.sum()}",
                # )
                reduce_to_target_dim_mat = reduce_to_target_dim_mat

                # reduce the features to target_feature_dim
                selected_features = selected_features @ reduce_to_target_dim_mat
                features =  features @ reduce_to_target_dim_mat
                C = features.shape[-1]

                # normalize the reduced features to [0, 1] along each dimension
                feat_min = features.reshape(-1, C).min(dim=0)[0]
                feat_max = features.reshape(-1, C).max(dim=0)[0]
                features = (features - feat_min) / (feat_max - feat_min)
                selected_features = (selected_features - feat_min) / (feat_max - feat_min)
                feat_min = feat_min.to(device)
                feat_max = feat_max.to(device)
                reduce_to_target_dim_mat = reduce_to_target_dim_mat.to(device)
            # we compute the first 3 principal components of the ViT features as the color
            reduction_mat, feat_color_min, feat_color_max = get_robust_pca(
                selected_features
            )
            # final features are of shape (num_imgs, num_patches_h, num_patches_w, target_feature_dim)
            features = features

            # save visualization parameters
            feat_dimension_reduction_mat = reduction_mat
            feat_color_min = feat_color_min
            feat_color_max = feat_color_max
            del temp_feats, selected_features

            # print(
            #     f"Feature PCA computed, shape: {feat_dimension_reduction_mat.shape}"
            # )
            # tensor: [91, 137, 64]
            x, y = torch.meshgrid(
                torch.arange(image.shape[1]),
                torch.arange(image.shape[0]),
                indexing="xy",
            )
            x, y = x.flatten(), y.flatten()
            x, y = x.to(device), y.to(device)

            # we compute the nearest DINO feature for each pixel
            # map (x, y) in the (W, H) space to (x * dino_scale[0], y * dino_scale[1]) in the (W//patch_size, H//patch_size) space
            dino_y = (y * featmap_downscale_factor[0]).long()
            dino_x = (x * featmap_downscale_factor[1]).long()
            # dino_feats are in CPU memory (because they are huge), so we need to move them to GPU
            features = features.squeeze()
            dino_feat = features[dino_y.cpu(), dino_x.cpu()]

            features = dino_feat.reshape(image.shape[0], image.shape[1], -1)
            feat_map = features.float()

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[0],
                        # for waymo
                        sky_mask=sky_mask, depth_map=depth_map, timestamp=time,
                        dynamic_mask=dynamic_mask, 
                        feat_map=feat_map, # [640,960,3]
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        intrinsic=intrinsic if load_intrinsic else None,
                        c2w=c2w if load_c2w else None,
                         ))
            
    return cam_infos

def readEmernerfInfo(args):

    eval = args.eval
    load_sky_mask = args.load_sky_mask
    load_panoptic_mask = args.load_panoptic_mask
    load_sam_mask = args.load_sam_mask
    load_dynamic_mask = args.load_dynamic_mask
    load_feat_map = args.load_feat_map
    load_intrinsic = args.load_intrinsic
    load_c2w = args.load_c2w
    save_occ_grid = args.save_occ_grid
    occ_voxel_size = args.occ_voxel_size
    recompute_occ_grid = args.recompute_occ_grid
    use_bg_gs = args.use_bg_gs
    white_background = args.white_background
    

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
    # modified from emer-nerf
    data_root = args.source_path
    image_folder = os.path.join(data_root, "images")
    num_seqs = len(os.listdir(image_folder))/5
    if end_time == -1:
        end_time = int(num_seqs)
    else:
        end_time += 1
        
    time_duration = args.time_duration
    time_interval = (time_duration[1] - time_duration[0]) / (end_time - start_time)
    
    camera_list = [1,0,2]
    truncated_min_range, truncated_max_range = -2, 80
    cam_frustum_range = [0.01, 80]
    # set img_list

    img_filepaths = []
    dynamic_mask_filepaths, sky_mask_filepaths = [], []
    semantic_mask_filepaths, instance_mask_filepaths = [], []
    sam_mask_filepaths = []
    feat_map_filepaths = []
    dynamic_mask_filepaths = []
    lidar_filepaths = []
    for t in range(start_time, end_time):
        for cam_idx in camera_list:
            img_filepaths.append(os.path.join(data_root, "images", f"{t:03d}_{cam_idx}.jpg"))
            #dynamic_mask_filepaths.append(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png"))
            sky_mask_filepaths.append(os.path.join(data_root, "sky_masks", f"{t:03d}_{cam_idx}.png"))
            #semantic_mask_filepaths.append(os.path.join(data_root, "semantic_masks", f"{t:03d}_{cam_idx}.png"))
            #instance_mask_filepaths.append(os.path.join(data_root, "instance_masks", f"{t:03d}_{cam_idx}.png"))
            if os.path.exists(os.path.join(data_root, "semantic_segs", f"{t:03d}_{cam_idx}.npy")):
                semantic_mask_filepaths.append(os.path.join(data_root, "semantic_segs", f"{t:03d}_{cam_idx}.npy"))
            else:
                semantic_mask_filepaths.append(None)
            if os.path.exists(os.path.join(data_root, "instance_segs", f"{t:03d}_{cam_idx}.npy")):
                instance_mask_filepaths.append(os.path.join(data_root, "instance_segs", f"{t:03d}_{cam_idx}.npy"))
            else:
                instance_mask_filepaths.append(None)
            if os.path.exists(os.path.join(data_root, "sam_masks", f"{t:03d}_{cam_idx}.jpg")):
                sam_mask_filepaths.append(os.path.join(data_root, "sam_masks", f"{t:03d}_{cam_idx}.jpg"))
            if os.path.exists(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png")):
                dynamic_mask_filepaths.append(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png"))
            if load_feat_map:
                feat_map_filepaths.append(os.path.join(data_root, "dinov2_vitb14", f"{t:03d}_{cam_idx}.npy"))
        lidar_filepaths.append(os.path.join(data_root, "lidar", f"{t:03d}.bin"))

    if load_feat_map:
        return_dict = extract_and_save_features(
                input_img_path_list=img_filepaths,
                saved_feat_path_list=feat_map_filepaths,
                img_shape=[644, 966],
                stride=7,
                model_type='dinov2_vitb14',
            )
    img_filepaths = np.array(img_filepaths)
    dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
    sky_mask_filepaths = np.array(sky_mask_filepaths)
    lidar_filepaths = np.array(lidar_filepaths)
    semantic_mask_filepaths = np.array(semantic_mask_filepaths)
    instance_mask_filepaths = np.array(instance_mask_filepaths)
    sam_mask_filepaths = np.array(sam_mask_filepaths)
    feat_map_filepaths = np.array(feat_map_filepaths)
    dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
    # ------------------
    # construct timestamps
    # ------------------
    # original_start_time = 0
    idx_list = range(original_start_time, end_time)
    # map time to [0,1]
    timestamp_mapper = {}
    time_line = [i for i in idx_list]
    time_length = end_time - original_start_time - 1
    for index, time in enumerate(time_line):
        timestamp_mapper[time] = (time-original_start_time)/time_length
    max_time = max(timestamp_mapper.values())
    # ------------------
    # load poses: intrinsic, c2w, l2w
    # ------------------
    _intrinsics = []
    cam_to_egos = []
    for i in range(len(camera_list)):
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
    # ------------------
    # get aabb: c2w --> frunstums --> aabb
    # ------------------
    # compute frustums
    frustums = []
    pix_corners = np.array( # load_size : [h, w]
        [[0,0],[0,load_size[0]],[load_size[1],load_size[0]],[load_size[1],0]]
    )
    for c2w, intri in zip(cam_to_worlds, intrinsics):
        frustum = []
        for cam_extent in cam_frustum_range:
            # pix_corners to cam_corners
            cam_corners = np.linalg.inv(intri) @ np.concatenate(
                [pix_corners, np.ones((4, 1))], axis=-1
            ).T * cam_extent
            # cam_corners to world_corners
            world_corners = c2w[:3, :3] @ cam_corners + c2w[:3, 3:4]
            # compute frustum
            frustum.append(world_corners)
        frustum = np.stack(frustum, axis=0)
        frustums.append(frustum)
    frustums = np.stack(frustums, axis=0)
    # compute aabb
    aabbs = []
    for frustum in frustums:
        flatten_frustum = frustum.transpose(0,2,1).reshape(-1,3)
        aabb_min = np.min(flatten_frustum, axis=0)
        aabb_max = np.max(flatten_frustum, axis=0)
        aabb = np.stack([aabb_min, aabb_max], axis=0)
        aabbs.append(aabb)
    aabbs = np.stack(aabbs, axis=0).reshape(-1,3)
    aabb = np.stack([np.min(aabbs, axis=0), np.max(aabbs, axis=0)], axis=0)
    print('cam frustum aabb min: ', aabb[0])
    print('cam frustum aabb max: ', aabb[1])
    # ------------------
    # get split: train and test splits from timestamps
    # ------------------
    # mask
    if stride != 0 :
        train_mask = (timestamps % int(stride) != 0) | (timestamps == 0)
    else:
        train_mask = np.ones(len(timestamps), dtype=bool)
    test_mask = ~train_mask
    # mask to index                                                                    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    full_idx = np.arange(len(timestamps))
    train_timestamps = timestamps[train_mask]
    test_timestamps = timestamps[test_mask]
    # ------------------
    # load points and depth map
    # ------------------
    pts_path = os.path.join(data_root, "lidar")
    load_lidar, load_depthmap = True, True
    depth_maps = None
    # bg-gs settings
    #use_bg_gs = False
    bg_scale = 2.0 # used to scale fg-aabb
    if not os.path.exists(pts_path) or not load_lidar:
        # random sample
        # Since this data set has no colmap data, we start with random points
        #num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")
        aabb_center = (aabb[0] + aabb[1]) / 2
        aabb_size = aabb[1] - aabb[0]
        # We create random points inside the bounds of the synthetic Blender scenes
        random_xyz = np.random.random((num_pts, 3)) 
        print('normed xyz min: ', np.min(random_xyz, axis=0))
        print('normed xyz max: ', np.max(random_xyz, axis=0))
        xyz = random_xyz * aabb_size + aabb[0]
        print('xyz min: ', np.min(xyz, axis=0))
        print('xyz max: ', np.max(xyz, axis=0))
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        # load lidar points
        origins, directions, points, ranges, laser_ids = [], [], [], [], []
        depth_maps = []
        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(0, len(lidar_filepaths), desc="loading lidar", dynamic_ncols=True):
            lidar_info = np.memmap(
                lidar_filepaths[t],
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 10) 
            #).reshape(-1, 14)
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
                lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + lidar_to_worlds[t][:3, 3:4]
            ).T
            lidar_points = (
                lidar_to_worlds[t][:3, :3] @ lidar_points.T
                + lidar_to_worlds[t][:3, 3:4]
            ).T
            if load_depthmap:
                # transform world-lidar to pixel-depth-map
                for cam_idx in range(len(camera_list)):
                    # world-lidar-pts --> camera-pts : w2c
                    c2w = cam_to_worlds[int(len(camera_list))*t + cam_idx]
                    w2c = np.linalg.inv(c2w)
                    cam_points = (
                        w2c[:3, :3] @ lidar_points.T
                        + w2c[:3, 3:4]
                    ).T
                    # camera-pts --> pixel-pts : intrinsic @ (x,y,z) = (u,v,1)*z
                    pixel_points = (
                        intrinsics[int(len(camera_list))*t + cam_idx] @ cam_points.T
                    ).T
                    # select points in front of the camera
                    pixel_points = pixel_points[pixel_points[:, 2]>0]
                    # normalize pixel points : (u,v,1)
                    image_points = pixel_points[:, :2] / pixel_points[:, 2:]
                    # filter out points outside the image
                    valid_mask = (
                        (image_points[:, 0] >= 0)
                        & (image_points[:, 0] < load_size[1])
                        & (image_points[:, 1] >= 0)
                        & (image_points[:, 1] < load_size[0])
                    )
                    pixel_points = pixel_points[valid_mask]     # pts_cam : (x,y,z)
                    image_points = image_points[valid_mask]     # pts_img : (u,v)
                    # compute depth map
                    depth_map = np.zeros(load_size)
                    depth_map[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)] = pixel_points[:, 2]
                    depth_maps.append(depth_map)
            # compute lidar directions
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = np.linalg.norm(lidar_directions, axis=-1, keepdims=True)
            lidar_directions = lidar_directions / lidar_ranges
            # time indices as timestamp
            #lidar_timestamps = np.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            points.append(lidar_points)
            ranges.append(lidar_ranges)
            laser_ids.append(lidar_ids)

        #origins = np.concatenate(origins, axis=0)
        #directions = np.concatenate(directions, axis=0)
        points = np.concatenate(points, axis=0)
        #ranges = np.concatenate(ranges, axis=0)
        #laser_ids = np.concatenate(laser_ids, axis=0)
        shs = np.random.random((len(points), 3)) / 255.0
        # filter points by cam_aabb 
        cam_aabb_mask = np.all((points >= aabb[0]) & (points <= aabb[1]), axis=-1)
        points = points[cam_aabb_mask]
        shs = shs[cam_aabb_mask]
        # construct occupancy grid to aid densification
        if save_occ_grid:
            #occ_grid_shape = (int(np.ceil((aabb[1, 0] - aabb[0, 0]) / occ_voxel_size)),
            #                    int(np.ceil((aabb[1, 1] - aabb[0, 1]) / occ_voxel_size)),
            #                    int(np.ceil((aabb[1, 2] - aabb[0, 2]) / occ_voxel_size)))
            if not os.path.exists(os.path.join(data_root, "occ_grid.npy")) or recompute_occ_grid:
                occ_grid = get_OccGrid(points, aabb, occ_voxel_size)
                np.save(os.path.join(data_root, "occ_grid.npy"), occ_grid)
            else:
                occ_grid = np.load(os.path.join(data_root, "occ_grid.npy"))
            print(f'Lidar points num : {len(points)}')
            print("occ_grid shape : ", occ_grid.shape)
            print(f'occ voxel num :{occ_grid.sum()} from {occ_grid.size} of ratio {occ_grid.sum()/occ_grid.size}')
        
        # downsample points
        points,shs = GridSample3D(points,shs)

        if len(points)>num_pts:
            downsampled_indices = np.random.choice(
                len(points), num_pts, replace=False
            )
            points = points[downsampled_indices]
            shs = shs[downsampled_indices]
        
        # check
        #voxel_coords = np.floor((points - aabb[0]) / occ_voxel_size).astype(int)
        #occ = occ_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
        #origins = origins[downsampled_indices] 
        
        ## 计算 points xyz 的范围
        xyz_min = np.min(points,axis=0)
        xyz_max = np.max(points,axis=0)
        print("init lidar xyz min:",xyz_min)
        print("init lidar xyz max:",xyz_max)        # lidar-points aabb (range)
        ## 设置 背景高斯点
        if use_bg_gs:
            fg_aabb_center, fg_aabb_size = (aabb[0] + aabb[1]) / 2, aabb[1] - aabb[0] # cam-frustum aabb
            # use bg_scale to scale the aabb
            bg_gs_aabb = np.stack([fg_aabb_center - fg_aabb_size * bg_scale / 2, 
                        fg_aabb_center + fg_aabb_size * bg_scale / 2], axis=0)
            bg_aabb_center, bg_aabb_size = (bg_gs_aabb[0] + bg_gs_aabb[1]) / 2, bg_gs_aabb[1] - bg_gs_aabb[0]
            # add bg_gs_aabb SURFACE points
            bg_points = sample_on_aabb_surface(bg_aabb_center, bg_aabb_size, 1000)
            print("bg_gs_points min:",np.min(bg_points,axis=0))
            print("bg_gs_points max:",np.max(bg_points,axis=0))
            # DO NOT add bg_gs_points to points
            #points = np.concatenate([points, bg_points], axis=0)
            #shs = np.concatenate([shs, np.random.random((len(bg_points), 3)) / 255.0], axis=0)
            bg_shs = np.random.random((len(bg_points), 3)) / 255.0
            # visualize
            #from utils.general_utils import visualize_points
            #visualize_points(points, fg_aabb_center, fg_aabb_size)
        # save ply
        ply_path = os.path.join(data_root, "points3d.ply")
        storePly(ply_path, points, SH2RGB(shs) * 255)
        pcd = BasicPointCloud(points=points, colors=SH2RGB(shs), normals=np.zeros((len(points), 3)))  
        if use_bg_gs:
            bg_ply_path = os.path.join(data_root, "bg-points3d.ply")
            storePly(bg_ply_path, bg_points, SH2RGB(bg_shs) * 255)
            bg_pcd = BasicPointCloud(points=bg_points, colors=SH2RGB(bg_shs), normals=np.zeros((len(bg_points), 3)))
        else:
            bg_pcd, bg_ply_path = None, None
        # load depth maps
        if load_depthmap:
            assert depth_maps is not None, "should not use random-init-gs, ans set load_depthmap=True"
            depth_maps = np.stack(depth_maps, axis=0)
    # ------------------
    # prepare cam-pose dict
    # ------------------
    train_frames_list = [] # time, transform_matrix(c2w), img_path
    test_frames_list = []
    full_frames_list = []
    for idx, t in enumerate(train_timestamps):
        frame_dict = dict(  time = time_line[t+start_time-original_start_time],   # 保存 相对帧索引
                            transform_matrix = cam_to_worlds[train_idx[idx]],
                            file_path = img_filepaths[train_idx[idx]],
                            intrinsic = intrinsics[train_idx[idx]],
                            load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                            sky_mask_path = sky_mask_filepaths[train_idx[idx]] if load_sky_mask else None,
                            depth_map = depth_maps[train_idx[idx]] if load_depthmap else None,
                            semantic_mask_path = semantic_mask_filepaths[train_idx[idx]] if load_panoptic_mask else None,
                            instance_mask_path = instance_mask_filepaths[train_idx[idx]] if load_panoptic_mask else None,
                            sam_mask_path = sam_mask_filepaths[train_idx[idx]] if load_sam_mask else None,
                            feat_map_path = feat_map_filepaths[train_idx[idx]] if load_feat_map else None,
                            dynamic_mask_path = dynamic_mask_filepaths[train_idx[idx]] if load_dynamic_mask else None,
        )
        train_frames_list.append(frame_dict)
    for idx, t in enumerate(test_timestamps):
        frame_dict = dict(  time = time_line[t+start_time-original_start_time],   # 保存 相对帧索引 
                            transform_matrix = cam_to_worlds[test_idx[idx]],
                            file_path = img_filepaths[test_idx[idx]],
                            intrinsic = intrinsics[test_idx[idx]],
                            load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                            sky_mask_path = sky_mask_filepaths[test_idx[idx]] if load_sky_mask else None,
                            depth_map = depth_maps[test_idx[idx]] if load_depthmap else None,
                            semantic_mask_path = semantic_mask_filepaths[test_idx[idx]] if load_panoptic_mask else None,
                            instance_mask_path = instance_mask_filepaths[test_idx[idx]] if load_panoptic_mask else None,
                            sam_mask_path = sam_mask_filepaths[test_idx[idx]] if load_sam_mask else None,
                            feat_map_path = feat_map_filepaths[test_idx[idx]] if load_feat_map else None,
                            dynamic_mask_path = dynamic_mask_filepaths[test_idx[idx]] if load_dynamic_mask else None,
        )
        test_frames_list.append(frame_dict)
    if len(test_timestamps)==0:
        full_frames_list = train_frames_list
    else:
        for idx, t in enumerate(timestamps):
            frame_dict = dict(  time = time_line[t+start_time-original_start_time],   # 保存 相对帧索引 
                                transform_matrix = cam_to_worlds[full_idx[idx]],
                                file_path = img_filepaths[full_idx[idx]],
                                intrinsic = intrinsics[full_idx[idx]],
                                load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                                sky_mask_path = sky_mask_filepaths[full_idx[idx]] if load_sky_mask else None,
                                depth_map = depth_maps[full_idx[idx]] if load_depthmap else None,
                                semantic_mask_path = semantic_mask_filepaths[full_idx[idx]] if load_panoptic_mask else None,
                                instance_mask_path = instance_mask_filepaths[full_idx[idx]] if load_panoptic_mask else None,
                                sam_mask_path = sam_mask_filepaths[full_idx[idx]] if load_sam_mask else None,
                                feat_map_path = feat_map_filepaths[full_idx[idx]] if load_feat_map else None,
                                dynamic_mask_path = dynamic_mask_filepaths[full_idx[idx]] if load_dynamic_mask else None,
            )
            full_frames_list.append(frame_dict)
    
    # ------------------
    # load cam infos: image, c2w, intrinsic, load_size
    # ------------------
    print("Reading Training Transforms")
    train_cam_infos = constructCameras_waymo(train_frames_list, white_background, timestamp_mapper, 
                                             load_intrinsic=load_intrinsic, load_c2w=load_c2w,start_time=start_time,original_start_time=original_start_time)
    print("Reading Test Transforms")
    test_cam_infos = constructCameras_waymo(test_frames_list, white_background, timestamp_mapper,
                                            load_intrinsic=load_intrinsic, load_c2w=load_c2w,start_time=start_time,original_start_time=original_start_time)
    print("Reading Full Transforms")
    full_cam_infos = constructCameras_waymo(full_frames_list, white_background, timestamp_mapper,
                                            load_intrinsic=load_intrinsic, load_c2w=load_c2w,start_time=start_time,original_start_time=original_start_time)
    # full_cam_infos = train_cam_infos
    
    #print("Generating Video Transforms")
    #video_cam_infos = generateCamerasFromTransforms_waymo(test_frames_list, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    #? waymo_loader
    # nerf_normalization['radius'] = 1/nerf_normalization['radius']

    # ------------------
    # find panoptic-objec numbers
    # ------------------
    num_panoptic_objects = 0
    panoptic_object_ids = None
    panoptic_id_to_idx = {}
    if load_panoptic_mask:
        panoptic_object_ids_list = []
        for cam in train_cam_infos+test_cam_infos:
            if cam.semantic_mask is not None and cam.instance_mask is not None:
                panoptic_object_ids = get_panoptic_id(cam.semantic_mask, cam.instance_mask).unique()
                panoptic_object_ids_list.append(panoptic_object_ids)
        # get unique panoptic_objects_ids
        panoptic_object_ids = torch.cat(panoptic_object_ids_list).unique().sort()[0].tolist()
        num_panoptic_objects = len(panoptic_object_ids)
        # map panoptic_id to idx
        for idx, panoptic_id in enumerate(panoptic_object_ids):
            panoptic_id_to_idx[panoptic_id] = idx

    scene_info = SceneInfo(point_cloud=pcd,
                           bg_point_cloud=bg_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           full_cameras=full_cam_infos,
                           #video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           # background settings
                           ply_path=ply_path,
                           bg_ply_path=bg_ply_path,
                           cam_frustum_aabb=aabb,
                           # panoptic segs
                           num_panoptic_objects=num_panoptic_objects,
                           panoptic_object_ids=panoptic_object_ids,
                           panoptic_id_to_idx=panoptic_id_to_idx,
                           # occ grid
                           occ_grid=occ_grid if save_occ_grid else None,
                           time_interval=time_interval,
                           time_duration=time_duration,
                           )

    return scene_info
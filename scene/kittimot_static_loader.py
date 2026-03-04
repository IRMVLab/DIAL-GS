import os
import cv2

import imageio
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.camera_utils import loadCam
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
camera_ls = [2, 3]

# camera_ls = [2]
"""
Most function brought from MARS
https://github.com/OPEN-AIR-SUN/mars/blob/69b9bf9d992e6b9f4027dfdc2a741c2a33eef174/mars/data/mars_kitti_dataparser.py
"""

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

def kitti_string_to_float(str):
    return float(str.split("e")[0]) * 10 ** int(str.split("e")[1])


def get_rotation(roll, pitch, heading):
    s_heading = np.sin(heading)
    c_heading = np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

    s_pitch = np.sin(pitch)
    c_pitch = np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    s_roll = np.sin(roll)
    c_roll = np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

    return rot


def tracking_calib_from_txt(calibration_path):
    """
    Extract tracking calibration information from a KITTI tracking calibration file.

    This function reads a KITTI tracking calibration file and extracts the relevant
    calibration information, including projection matrices and transformation matrices
    for camera, LiDAR, and IMU coordinate systems.

    Args:
        calibration_path (str): Path to the KITTI tracking calibration file.

    Returns:
        dict: A dictionary containing the following calibration information:
            P0, P1, P2, P3 (np.array): 3x4 projection matrices for the cameras.
            Tr_cam2camrect (np.array): 4x4 transformation matrix from camera to rectified camera coordinates.
            Tr_velo2cam (np.array): 4x4 transformation matrix from LiDAR to camera coordinates.
            Tr_imu2velo (np.array): 4x4 transformation matrix from IMU to LiDAR coordinates.
    """
    # Read the calibration file
    f = open(calibration_path)
    calib_str = f.read().splitlines()

    # Process the calibration data
    calibs = []
    for calibration in calib_str:
        calibs.append(np.array([kitti_string_to_float(val) for val in calibration.split()[1:]]))

    # Extract the projection matrices
    P0 = np.reshape(calibs[0], [3, 4])
    P1 = np.reshape(calibs[1], [3, 4])
    P2 = np.reshape(calibs[2], [3, 4])
    P3 = np.reshape(calibs[3], [3, 4])

    # Extract the transformation matrix for camera to rectified camera coordinates
    Tr_cam2camrect = np.eye(4)
    R_rect = np.reshape(calibs[4], [3, 3])
    Tr_cam2camrect[:3, :3] = R_rect

    # Extract the transformation matrices for LiDAR to camera and IMU to LiDAR coordinates
    Tr_velo2cam = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    Tr_imu2velo = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    return {
        # rectified cam2 -> pixel plane
        "P0": P0,
        "P1": P1,
        "P2": P2, # 标准相机，很多变换都是基于此相机
        "P3": P3,
        # 用于对齐相机的极线矫正矩阵
        "Tr_cam2camrect": Tr_cam2camrect, 
        # 激光雷达到2号相机的变换矩阵
        "Tr_velo2cam": Tr_velo2cam,
        # IMU到激光雷达的变换矩阵
        "Tr_imu2velo": Tr_imu2velo,
    }


def calib_from_txt(calibration_path):
    """
    Read the calibration files and extract the required transformation matrices and focal length.

    Args:
        calibration_path (str): The path to the directory containing the calibration files.

    Returns:
        tuple: A tuple containing the following elements:
            traimu2v (np.array): 4x4 transformation matrix from IMU to Velodyne coordinates.
            v2c (np.array): 4x4 transformation matrix from Velodyne to left camera coordinates.
            c2leftRGB (np.array): 4x4 transformation matrix from left camera to rectified left camera coordinates.
            c2rightRGB (np.array): 4x4 transformation matrix from right camera to rectified right camera coordinates.
            focal (float): Focal length of the left camera.
    """
    c2c = []

    # Read and parse the camera-to-camera calibration file
    f = open(os.path.join(calibration_path, "calib_cam_to_cam.txt"), "r")
    cam_to_cam_str = f.read()
    [left_cam, right_cam] = cam_to_cam_str.split("S_02: ")[1].split("S_03: ")
    cam_to_cam_ls = [left_cam, right_cam]

    # Extract the transformation matrices for left and right cameras
    for i, cam_str in enumerate(cam_to_cam_ls):
        r_str, t_str = cam_str.split("R_0" + str(i + 2) + ": ")[1].split("\nT_0" + str(i + 2) + ": ")
        t_str = t_str.split("\n")[0]
        R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
        R = np.reshape(R, [3, 3])
        t = np.array([kitti_string_to_float(t) for t in t_str.split(" ")])
        Tr = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

        t_str_rect, s_rect_part = cam_str.split("\nT_0" + str(i + 2) + ": ")[1].split("\nS_rect_0" + str(i + 2) + ": ")
        s_rect_str, r_rect_part = s_rect_part.split("\nR_rect_0" + str(i + 2) + ": ")
        r_rect_str = r_rect_part.split("\nP_rect_0" + str(i + 2) + ": ")[0]
        R_rect = np.array([kitti_string_to_float(r) for r in r_rect_str.split(" ")])
        R_rect = np.reshape(R_rect, [3, 3])
        t_rect = np.array([kitti_string_to_float(t) for t in t_str_rect.split(" ")])
        Tr_rect = np.concatenate(
            [np.concatenate([R_rect, t_rect[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]]
        )

        c2c.append(Tr_rect)

    c2leftRGB = c2c[0]
    c2rightRGB = c2c[1]

    # Read and parse the Velodyne-to-camera calibration file
    f = open(os.path.join(calibration_path, "calib_velo_to_cam.txt"), "r")
    velo_to_cam_str = f.read()
    r_str, t_str = velo_to_cam_str.split("R: ")[1].split("\nT: ")
    t_str = t_str.split("\n")[0]
    R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(" ")])
    v2c = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

    # Read and parse the IMU-to-Velodyne calibration file
    f = open(os.path.join(calibration_path, "calib_imu_to_velo.txt"), "r")
    imu_to_velo_str = f.read()
    r_str, t_str = imu_to_velo_str.split("R: ")[1].split("\nT: ")
    R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(" ")])
    imu2v = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

    # Extract the focal length of the left camera
    focal = kitti_string_to_float(left_cam.split("P_rect_02: ")[1].split()[0])

    return imu2v, v2c, c2leftRGB, c2rightRGB, focal


def get_poses_calibration(basedir, oxts_path_tracking=None, selected_frames=None):
    """
    Extract poses and calibration information from the KITTI dataset.

    This function processes the OXTS data (GPS/IMU) and extracts the
    pose information (translation and rotation) for each frame. It also
    retrieves the calibration information (transformation matrices and focal length)
    required for further processing.

    Args:
        basedir (str): The base directory containing the KITTI dataset.
        oxts_path_tracking (str, optional): Path to the OXTS data file for tracking sequences.
            If not provided, the function will look for OXTS data in the basedir.
        selected_frames (list, optional): A list of frame indices to process.
            If not provided, all frames in the dataset will be processed.

    Returns:
        tuple: A tuple containing the following elements:
            poses (np.array): An array of 4x4 pose matrices representing the vehicle's
                position and orientation for each frame (IMU pose).
            calibrations (dict): A dictionary containing the transformation matrices
                and focal length obtained from the calibration files.
            focal (float): The focal length of the left camera.
    """

    def oxts_to_pose(oxts):
        """
        OXTS (Oxford Technical Solutions) data typically refers to the data generated by an Inertial and GPS Navigation System (INS/GPS) that is used to provide accurate position, orientation, and velocity information for a moving platform, such as a vehicle. In the context of the KITTI dataset, OXTS data is used to provide the ground truth for the vehicle's trajectory and 6 degrees of freedom (6-DoF) motion, which is essential for evaluating and benchmarking various computer vision and robotics algorithms, such as visual odometry, SLAM, and object detection.

        The OXTS data contains several important measurements:

        1. Latitude, longitude, and altitude: These are the global coordinates of the moving platform.
        2. Roll, pitch, and yaw (heading): These are the orientation angles of the platform, usually given in Euler angles.
        3. Velocity (north, east, and down): These are the linear velocities of the platform in the local navigation frame.
        4. Accelerations (ax, ay, az): These are the linear accelerations in the platform's body frame.
        5. Angular rates (wx, wy, wz): These are the angular rates (also known as angular velocities) of the platform in its body frame.

        In the KITTI dataset, the OXTS data is stored as plain text files with each line corresponding to a timestamp. Each line in the file contains the aforementioned measurements, which are used to compute the ground truth trajectory and 6-DoF motion of the vehicle. This information can be further used for calibration, data synchronization, and performance evaluation of various algorithms.
        """
        poses = []

        def latlon_to_mercator(lat, lon, s):
            """
            Converts latitude and longitude coordinates to Mercator coordinates (x, y) using the given scale factor.

            The Mercator projection is a widely used cylindrical map projection that represents the Earth's surface
            as a flat, rectangular grid, distorting the size of geographical features in higher latitudes.
            This function uses the scale factor 's' to control the amount of distortion in the projection.

            Args:
                lat (float): Latitude in degrees, range: -90 to 90.
                lon (float): Longitude in degrees, range: -180 to 180.
                s (float): Scale factor, typically the cosine of the reference latitude.

            Returns:
                list: A list containing the Mercator coordinates [x, y] in meters.
            """
            r = 6378137.0  # the Earth's equatorial radius in meters
            x = s * r * ((np.pi * lon) / 180)
            y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
            return [x, y]

        # Compute the initial scale and pose based on the selected frames
        if selected_frames is None:
            lat0 = oxts[0][0]
            scale = np.cos(lat0 * np.pi / 180)
            pose_0_inv = None
        else:
            oxts0 = oxts[selected_frames[0][0]]
            lat0 = oxts0[0]
            scale = np.cos(lat0 * np.pi / 180)

            pose_i = np.eye(4)

            [x, y] = latlon_to_mercator(oxts0[0], oxts0[1], scale)
            z = oxts0[2]
            translation = np.array([x, y, z])
            rotation = get_rotation(oxts0[3], oxts0[4], oxts0[5])
            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

        # Iterate through the OXTS data and compute the corresponding pose matrices
        for oxts_val in oxts:
            pose_i = np.zeros([4, 4])
            pose_i[3, 3] = 1

            [x, y] = latlon_to_mercator(oxts_val[0], oxts_val[1], scale)
            z = oxts_val[2]
            translation = np.array([x, y, z])

            roll = oxts_val[3]
            pitch = oxts_val[4]
            heading = oxts_val[5]
            rotation = get_rotation(roll, pitch, heading)  # (3,3)

            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)  # (4, 4)
            if pose_0_inv is None:
                pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

            pose_i = np.matmul(pose_0_inv, pose_i)
            poses.append(pose_i)

        return np.array(poses)

    # If there is no tracking path specified, use the default path
    if oxts_path_tracking is None:
        oxts_path = os.path.join(basedir, "oxts/data")
        oxts = np.array([np.loadtxt(os.path.join(oxts_path, file)) for file in sorted(os.listdir(oxts_path))])
        calibration_path = os.path.dirname(basedir)

        calibrations = calib_from_txt(calibration_path)

        focal = calibrations[4]

        poses = oxts_to_pose(oxts)

    # If a tracking path is specified, use it to load OXTS data and compute the poses
    else:
        oxts_tracking = np.loadtxt(oxts_path_tracking)
        poses = oxts_to_pose(oxts_tracking)  # (n_frames, 4, 4)
        calibrations = None
        focal = None
        # Set velodyne close to z = 0
        # poses[:, 2, 3] -= 0.8

    # Return the poses, calibrations, and focal length
    return poses, calibrations, focal


def invert_transformation(rot, t):
    # 手动构造逆变换矩阵
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0.0, 0.0, 0.0, 1.0]])])


def get_camera_poses_tracking(poses_velo_w_tracking, tracking_calibration, selected_frames, scene_no=None):
    exp = False
    camera_poses = []

    opengl2kitti = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    start_frame = selected_frames[0]
    end_frame = selected_frames[1]

    #####################
    # Debug Camera offset
    if scene_no == 2:
        yaw = np.deg2rad(0.7)  ## Affects camera rig roll: High --> counterclockwise
        pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(0.9)  ## Affects camera rig pitch: High -->  up
        # roll = np.deg2rad(1.2)
    elif scene_no == 1:
        if exp:
            yaw = np.deg2rad(0.3)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.6)  ## Affects camera rig yaw: High --> Turn Right
            # pitch = np.deg2rad(-0.97)
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
            # roll = np.deg2rad(1.2)
        else:
            yaw = np.deg2rad(0.5)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
    else:
        yaw = np.deg2rad(0.05)
        pitch = np.deg2rad(-0.75)
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(1.05)
        # roll = np.deg2rad(1.2)

    cam_debug = np.eye(4)
    cam_debug[:3, :3] = get_rotation(roll, pitch, yaw)

    Tr_cam2camrect = tracking_calibration["Tr_cam2camrect"]
    Tr_cam2camrect = np.matmul(Tr_cam2camrect, cam_debug)
    Tr_camrect2cam = invert_transformation(Tr_cam2camrect[:3, :3], Tr_cam2camrect[:3, 3])
    Tr_velo2cam = tracking_calibration["Tr_velo2cam"]
    Tr_cam2velo = invert_transformation(Tr_velo2cam[:3, :3], Tr_velo2cam[:3, 3])

    camera_poses_imu = []
    for cam in camera_ls:
        Tr_camrect2cam_i = tracking_calibration["Tr_camrect2cam0" + str(cam)]
        Tr_cam_i2camrect = invert_transformation(Tr_camrect2cam_i[:3, :3], Tr_camrect2cam_i[:3, 3])
        # transform camera axis from kitti to opengl for nerf:
        cam_i_camrect = np.matmul(Tr_cam_i2camrect, opengl2kitti)
        cam_i_cam0 = np.matmul(Tr_camrect2cam, cam_i_camrect)
        cam_i_velo = np.matmul(Tr_cam2velo, cam_i_cam0)

        cam_i_w = np.matmul(poses_velo_w_tracking, cam_i_velo) # (N, 4, 4)
        camera_poses_imu.append(cam_i_w)

    for i, cam in enumerate(camera_ls):
        for frame_no in range(start_frame, end_frame + 1):
            camera_poses.append(camera_poses_imu[i][frame_no])

    return np.array(camera_poses) # (cam_num*N , 4, 4)


def get_scene_images_tracking(cam_num, tracking_path, sequence, selected_frames):
    [start_frame, end_frame] = selected_frames
    # [[cam02_f0,cam02_f1...],[cam03_f0,cam03_f1...]]

    img_name = [] 
    sky_name = []
    semantic_mask_name = []
    tracking_data_name = []
    
    for i in range(cam_num):
        img_name.append([])
        sky_name.append([])
        semantic_mask_name.append([])
        tracking_data_name.append([])
    
        
        img_path = os.path.join(os.path.join(tracking_path, f"image_0{2+i}"), sequence)
        # right_img_path = os.path.join(os.path.join(tracking_path, "image_03"), sequence)

        sky_path = os.path.join(os.path.join(tracking_path, f"sky_0{2+i}"), sequence)
        # right_sky_path = os.path.join(os.path.join(tracking_path, "sky_03"), sequence)

        semantic_mask_path = os.path.join(os.path.join(tracking_path, f"semantic_mask_0{2+i}"), sequence)
        # right_semantic_mask_path = os.path.join(os.path.join(tracking_path, "semantic_mask_03"), sequence)

        tracking_data_path = os.path.join(os.path.join(tracking_path, f"tracking_data_0{2+i}"), sequence)
        # right_tracking_data_path = os.path.join(os.path.join(tracking_path, "tracking_data_03"), sequence)


        for frame_no in range(len(os.listdir(img_path))):
            if start_frame <= frame_no <= end_frame:
                base_name = str(frame_no).zfill(6) 
                img_name[i].append(os.path.join(img_path, base_name+".png"))
                # img_name[1].append(os.path.join(right_img_path, base_name+".png"))
                
                sky_name[i].append(os.path.join(sky_path, base_name+".png"))
                # sky_name[1].append(os.path.join(right_sky_path, base_name+".png"))

                semantic_mask_name[i].append(os.path.join(semantic_mask_path, base_name+".png"))
                # semantic_mask_name[1].append(os.path.join(right_semantic_mask_path, base_name+".png"))
                
                # TODO: 改为接入dyanmaic_mask和 id_mask而非tracking_data
                tracking_data_name[i].append(os.path.join(tracking_data_path, base_name+".npy"))
                # tracking_data_name[1].append(os.path.join(right_tracking_data_path, base_name+".npy"))



    return img_name, sky_name, semantic_mask_name, tracking_data_name

def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))

def auto_orient_and_center_poses(
    poses,
):
    """
    From nerfstudio
    https://github.com/nerfstudio-project/nerfstudio/blob/8e0c68754b2c440e2d83864fac586cddcac52dc4/nerfstudio/cameras/camera_utils.py#L515
    """
    origins = poses[..., :3, 3]
    mean_origin = torch.mean(origins, dim=0)
    translation = mean_origin
    up = torch.mean(poses[:, :3, 1], dim=0)
    up = up / torch.linalg.norm(up)
    rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
    transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
    oriented_poses = transform @ poses
    return oriented_poses, transform


def visualize_point_cloud_projection(point_camera, image, fx, fy, cx, cy, H, W, save_path, alpha=0.7, depth_colormap=cv2.COLORMAP_JET):
    """
    将点云投影到图像平面，生成伪深度图并与RGB图像叠加
    
    Args:
        point_camera: 相机坐标系下的点云 (N, 3)
        image: RGB图像 (H, W, 3), 值范围[0,1]
        fx, fy, cx, cy: 相机内参
        H, W: 图像高度和宽度
        save_path: 保存路径
        alpha: 深度图透明度
        depth_colormap: 深度图颜色映射
    """
    # 过滤掉相机后方的点
    valid_points = point_camera[:, 2] > 0.1
    points_valid = point_camera[valid_points]
    
    if len(points_valid) == 0:
        print("No valid points to project")
        return
    
    # 投影到像素坐标
    pixels_x = (points_valid[:, 0] / points_valid[:, 2] * fx + cx).astype(np.int32)
    pixels_y = (points_valid[:, 1] / points_valid[:, 2] * fy + cy).astype(np.int32)
    depths = points_valid[:, 2]
    
    # 过滤出在图像范围内的点
    valid_pixels = (pixels_x >= 0) & (pixels_x < W) & (pixels_y >= 0) & (pixels_y < H)
    pixels_x = pixels_x[valid_pixels]
    pixels_y = pixels_y[valid_pixels]
    depths = depths[valid_pixels]
    
    if len(depths) == 0:
        print("No points project into image")
        return
        
    # 创建深度图
    depth_map = np.zeros((H, W), dtype=np.float32)
    
    # 对于重叠的像素，取最近的深度值
    for i in range(len(pixels_x)):
        x, y, d = pixels_x[i], pixels_y[i], depths[i]
        if depth_map[y, x] == 0 or d < depth_map[y, x]:
            depth_map[y, x] = d
    
    # 归一化深度图到[0,255]
    depth_min, depth_max = depths.min(), depths.max()
    depth_normalized = np.zeros_like(depth_map)
    mask = depth_map > 0
    depth_normalized[mask] = ((depth_map[mask] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    
    # 应用颜色映射
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), depth_colormap)
    depth_colored = depth_colored.astype(np.float32) / 255.0
    
    # 创建掩码，只在有深度值的地方叠加
    depth_mask = (depth_map > 0)[..., None]
    
    # 将图像转换为0-1范围（如果还不是的话）
    image_vis = image.copy()
    
    # 叠加深度图和RGB图像
    overlay = np.where(depth_mask, 
                      alpha * depth_colored + (1 - alpha) * image_vis,
                      image_vis)
    
    # 保存结果
    plt.figure(figsize=(15, 5))
    
    # 原始RGB图像
    plt.subplot(1, 3, 1)
    plt.imshow(image_vis)
    plt.title('Original RGB')
    plt.axis('off')
    
    # 深度图
    plt.subplot(1, 3, 2)
    plt.imshow(depth_normalized, cmap='jet')
    plt.title(f'Projected Depth Map\n(min: {depth_min:.2f}m, max: {depth_max:.2f}m)')
    plt.colorbar()
    plt.axis('off')
    
    # 叠加结果
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f'RGB + Depth Overlay (α={alpha})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    



class KittiDataset:
    def __init__(self, args):
        self.args = args
        self.cam_num = len(camera_ls)  # 2
        self.resolution_scales = args.resolution_scales # [1, 2, 4, 8, 16]
        self.scale_index = len(self.resolution_scales) - 1 # 从后往前索引，训练时从粗糙到精细
        self.cam_infos = [] # 相机信息列表 

        # [
        # [caminfo2, caminfo3], # frame0
        # [caminfo2, caminfo3], # frame1
        #     ...
        # ]
       
        self.Cameras = {} # 相机字典
        # {
        #     "scale1": [Cam2_f1,Cam2_f2,Cam2_f3,...,Cam2_fn, Cam3_f1,Cam3_f2,Cam3_f3,...,Cam3_fn], 
        #     "scale2": [Cam2_f1,Cam2_f2,Cam2_f3,...,Cam2_fn, Cam3_f1,Cam3_f2,Cam3_f3,...,Cam3_fn],
        #     "scale4": [Cam2_f1,Cam2_f2,Cam2_f3,...,Cam2_fn, Cam3_f1,Cam3_f2,Cam3_f3,...,Cam3_fn],
        # }
        self.pcd = None # 点云数据
        self.scale_factor = 1.0
        self.first_frame = args.start_frame # 65
        self.last_frame = args.end_frame # 120
        self.frame_num = self.last_frame-self.first_frame+1


    def load_cam_info(self):

        points_all = []
        # image_02 和 image_03是左右相机,编号为02与03
        basedir = self.args.source_path # /data/kitti_pvg/training/image_02/0001 
        scene_id = basedir[-4:]  # 0001 0002 0006
        kitti_scene_no = int(scene_id)
        tracking_path = basedir[:-13]  # /data/kitti_pvg/training/
        calibration_path = os.path.join(os.path.join(tracking_path, "calib"), scene_id + ".txt") # /data/kitti_pvg/training/calib/0001.txt
        oxts_path_tracking = os.path.join(os.path.join(tracking_path, "oxts"), scene_id + ".txt") # /data/kitti_pvg/training/oxts/0001.txt
 
        tracking_calibration = tracking_calib_from_txt(calibration_path)
        focal_X = tracking_calibration["P2"][0, 0]
        focal_Y = tracking_calibration["P2"][1, 1]
        poses_imu_w_tracking, _, _ = get_poses_calibration(basedir, oxts_path_tracking)  # (n_frames, 4, 4) imu pose （imu2world)

        tr_imu2velo = tracking_calibration["Tr_imu2velo"]
        tr_velo2imu = invert_transformation(tr_imu2velo[:3, :3], tr_imu2velo[:3, 3]) # (4, 4) velodyne to imu   
        poses_velo_w_tracking = np.matmul(poses_imu_w_tracking, tr_velo2imu)  # (n_frames, 4, 4) velodyne pose (velo2world)

        # Get camera Poses   
        for cam_i in range(self.cam_num):
            transformation = np.eye(4)
            projection = tracking_calibration["P" + str(cam_i + 2)]  # rectified camera coordinate system -> image
            K_inv = np.linalg.inv(projection[:3, :3])
            R_t = projection[:3, 3]
            t_crect2c = np.matmul(K_inv, R_t)
            transformation[:3, 3] = t_crect2c
            # 获得 Tr_camrect2cam02 Tr_camrect2cam03
            tracking_calibration["Tr_camrect2cam0" + str(cam_i + 2)] = transformation
        
        selected_frames = [self.first_frame, self.last_frame]
        sequ_frames = selected_frames

        cam_poses_tracking = get_camera_poses_tracking(
            poses_velo_w_tracking, tracking_calibration, sequ_frames, kitti_scene_no
        )
        poses_velo_w_tracking = poses_velo_w_tracking[self.first_frame:self.last_frame + 1]

        # Orients and centers the poses 将相机姿态进行自动对齐和中心化
        oriented = torch.from_numpy(np.array(cam_poses_tracking).astype(np.float32))  # (n_frames, 3, 4)
        oriented, transform_matrix = auto_orient_and_center_poses(
            oriented
        )  # oriented (n_frames, 3, 4), transform_matrix (3, 4)
        row = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        zeros = torch.zeros(oriented.shape[0], 1, 4)
        oriented = torch.cat([oriented, zeros], dim=1)
        oriented[:, -1] = row  # (n_frames, 4, 4)
        transform_matrix = torch.cat([transform_matrix, row[None, :]], dim=0)  # (4, 4)
        cam_poses_tracking = oriented.numpy()
        transform_matrix = transform_matrix.numpy()

        # 加载了所选帧的图像和天空mask路径
        image_filenames, sky_filenames, semantic_mask_filenames, tracking_data_filenames = \
               get_scene_images_tracking(self.cam_num, tracking_path, scene_id, sequ_frames)

        # Align Axis with vkitti axis 将KITTI坐标系转换为vkitti坐标系
        poses = cam_poses_tracking.astype(np.float32)
        poses[:, :, 1:3] *= -1

        test_load_image = imageio.imread(image_filenames[0][0])
        image_height, image_width = test_load_image.shape[:2]
        cx, cy = image_width / 2.0, image_height / 2.0
        poses[..., :3, 3] *= self.scale_factor
        c2ws = poses # cam_num*N, 4, 4 先是N个cam2的位姿，再是N个cam3的位姿
        c2ws = c2ws.reshape(self.cam_num, self.frame_num, 4, 4)  # (cam_num, N, 4, 4)

        # self.frame_num = 5
        for idx in tqdm(range(self.frame_num), desc="Loading data"):
            self.cam_infos.append([])  # 每个相机的每一帧都添加一个空列表
            
            for cam in range(self.cam_num):
                c2w = c2ws[cam, idx, :, :]  # (4, 4) camera to world transformation matrix
                # print(f"Load frame {idx}, camera {cam} with position {c2w[3, :3]}")
                w2c = np.linalg.inv(c2w)
                image_path = image_filenames[cam][idx]
                image_name = os.path.basename(image_path)[:-4]
                sky_path = sky_filenames[cam][idx]
                semantic_mask_path= semantic_mask_filenames[cam][idx]
                tracking_data_path = tracking_data_filenames[cam][idx]
                im_data = Image.open(image_path)
                W, H = im_data.size
                image = np.array(im_data) / 255.
                sky_data = cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)
                sky_mask = (sky_data > 0).astype(np.float32) # 1 表示天空，0表示非天空
                semantic_data = cv2.imread(semantic_mask_path, cv2.IMREAD_GRAYSCALE)
                semantic_mask = (semantic_data > 0).astype(np.float32)  # 1表示人车
                # 对semantic_mask进行形态学处理
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 30))  # 横宽窄，纵高高
                dilated = cv2.dilate(semantic_mask, kernel, iterations=1)
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
                semantic_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
                
                
                tracking_data = np.load(tracking_data_path) 

                # timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * (idx % (len(c2ws) // 2)) / (len(c2ws) // 2 - 1)
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                # if idx < len(c2ws) / 2:
                if cam == 0:
                    # 只在第一个相机时加载点云
                    point = np.fromfile(os.path.join(tracking_path, "velodyne", scene_id, image_name + ".bin"), dtype=np.float32).reshape(-1, 4)
                    point_xyz = point[:, :3]
                    point_xyz_world = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ poses_velo_w_tracking[idx].T)[:, :3]
                    points_all.append(point_xyz_world)
                    # frame_num = len(c2ws) // 2
                    # frame_num = len(c2ws)
                    # point_xyz = points[idx%frame_num]
                
                point_cur_frame = points_all[idx%self.frame_num] 
                point_camera = (np.pad(point_cur_frame, ((0, 0), (0, 1)), constant_values=1) @ transform_matrix.T @ w2c.T)[:, :3]*self.scale_factor


                # print("Max depth of point cloud: ", np.max(point_camera[:, 2]))
                # print("Min depth of point cloud: ", np.min(point_camera[:, 2]))
                
                # self.cam_infos[idx].append(CameraInfo(uid=idx * , R=R, T=T,
                #                             image=image,
                #                             image_path=image_filenames[idx], image_name=image_filenames[idx],
                #                             width=W, height=H, 
                #                             fx=focal_X, fy=focal_Y, cx=cx, cy=cy, sky_mask=sky_mask,
                #                             pointcloud_camera=point_camera))
                
                self.cam_infos[idx].append(CameraInfo(uid=idx * self.cam_num + cam, R=R, T=T,
                                            image=image, 
                                            image_path=image_path, image_name=image_name,
                                            width= W, height= H, 
                                            pointcloud_camera = point_camera,
                                            fx=focal_X, fy=focal_Y, cx=cx, cy=cy,
                                            sky_mask=sky_mask, 
                                            semantic_mask=semantic_mask,
                                            normal_map=None))
                # print(f"ADD caminfo with position {T} and rotation:\n{R}")
                
                # 添加可视化（只对前几帧和第一个相机进行可视化，避免生成太多图片）

                vis_save_dir = os.path.join(self.args.model_path, "lidar_projection_vis")
                os.makedirs(vis_save_dir, exist_ok=True)
                vis_save_path = os.path.join(vis_save_dir, f"frame_{idx:03d}_cam_{cam:02d}.png")
                
                visualize_point_cloud_projection(
                    point_camera=point_camera,
                    image=image,
                    fx=focal_X, fy=focal_Y, cx=cx, cy=cy,
                    H=H, W=W,
                    save_path=vis_save_path,
                    alpha=0.9
                )
               

                
            
        points_all = np.concatenate(points_all, axis=0)
        points_all = (np.concatenate([points_all, np.ones_like(points_all[:,:1])], axis=-1) @ transform_matrix.T)[:, :3]
        indices = np.random.choice(points_all.shape[0], self.args.num_pts, replace=True)
        points_all = points_all[indices]
        self.pcd = points_all

        print(f"############# Load {self.pcd.shape[0]} points #############")
         
        # PCA 变换
        w2cs = np.zeros((self.frame_num * self.cam_num, 4, 4))
        Rs = []
        Ts = []
        print("Length of cam_infos: ", len(self.cam_infos))
        for frame_id in range(self.frame_num):
            for cam_id in range(self.cam_num):
                Rs.append(self.cam_infos[frame_id][cam_id].R)
                Ts.append(self.cam_infos[frame_id][cam_id].T)
        Rs = np.stack(Rs, axis=0) # (frame_num * cam_num, 3, 3)
        Ts = np.stack(Ts, axis=0) # (frame_num * cam_num, 3)
        print("Rs shape: ", Rs.shape, "Ts shape: ", Ts.shape)
        print("Rs: ", Rs[:3], "Ts: ", Ts[:3])

        w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
        w2cs[:, :3, 3] = Ts
        w2cs[:, 3, 3] = 1
        c2ws = unpad_poses(np.linalg.inv(w2cs))
        c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=self.args.fix_radius)
        print("PCA transform matrix:\n", transform)
        c2ws = pad_poses(c2ws)
        for frame_id in range(self.frame_num):
            for cam_id in range(self.cam_num):
                idx = frame_id * self.cam_num + cam_id
                cam_info = self.cam_infos[frame_id][cam_id]
                c2w = c2ws[idx]
                w2c = np.linalg.inv(c2w)
                cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                cam_info.T[:] = w2c[:3, 3]
                cam_info.pointcloud_camera[:] *= scale_factor
        
        self.pcd = (np.pad(self.pcd, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3] 

        
    def load_Cam(self):
        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras at resolution scale {}".format(resolution_scale))
            self.Cameras[f"scale{resolution_scale}"] = []
            for idx in tqdm(range(self.frame_num), desc=f"Loading data at different scales"):
                self.Cameras[f"scale{resolution_scale}"].append([])
                for cam_j in range(self.cam_num):
                    cam_info = self.cam_infos[idx][cam_j]
                    uid = f"id{idx}_scale{resolution_scale}_cam{cam_j}"
                    # print(f"Loading image {cam_info.image_name}  with pose {cam_info.T}")
                    self.Cameras["scale"+str(resolution_scale)][idx].append(loadCam(self.args, uid, cam_info, resolution_scale))

    def load(self):
        self.load_cam_info()   # 加载相机信息
        flattened_cam_infos = [cam for sublist in self.cam_infos for cam in sublist] # 将相机信息列表展平
        nerf_normalization = getNerfppNorm(flattened_cam_infos) # 获取NeRF的归一化参数
        nerf_normalization['radius'] = 1
        self.cameras_extent = nerf_normalization['radius'] # 计算相机的包围盒大小
        self.load_Cam()
    
    
    def getTestCameras(self):
        return None

    
    def __len__(self):
        return self.frame_num * self.cam_num 

    
    def get_cam(self, scale, frame_idx, cam_idx):

        return self.Cameras[f"scale{scale}"][frame_idx][cam_idx]



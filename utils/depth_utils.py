from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def gt_depth_generator(source_path, cam_num):
    """
    Generate ground truth depth from the given parameters.
    """
    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(source_path, "calib"))) if f.endswith('.txt')]
    frame_num = len(car_list)
    load_size = [640, 960]
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    # point_xyz = [] # 所有点云的xyz坐标
    
    # for car_id in car_list:
    #     point = np.fromfile(os.path.join(source_path, "velodyne", car_id + ".bin"),
    #                         dtype=np.float32, count=-1).reshape(-1, 6)
    #     point_xyz.append(point[:, :3])  # 只取xyz坐标
    
    # point_xyz = np.concatenate(point_xyz, axis=0)  # (N, 3)

    for idx, car_id in tqdm(enumerate(car_list), desc="Generating GT depth", total=frame_num):
        # ego_pose => lidar2world
        ego_pose = np.loadtxt(os.path.join(source_path, 'pose', car_id + '.txt')) # (4, 4)

        # CAMERA DIRECTION: RIGHT DOWN FORWARDS
        with open(os.path.join(source_path, 'calib', car_id + '.txt')) as f:
            calib_data = f.readlines()
            L = [list(map(float, line.split()[1:])) for line in calib_data]
        
        Ks = np.array(L[:5]).reshape(-1, 3, 4)[:, :, :3] # 相机内参矩阵 (5, 3, 4) => (5, 3, 3)
        lidar2cam = np.array(L[-5:]).reshape(-1, 3, 4) # (5, 3, 4) 
        lidar2cam = pad_poses(lidar2cam) # (5, 4, 4)

        # 读取该帧以及之后的每隔5帧点云（直到最后一帧）
        point = np.fromfile(os.path.join(source_path, "velodyne", car_id + ".bin"),
                            dtype=np.float32, count=-1).reshape(-1, 6) # count=-1表示读取所有数据
        point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1) 
        # point_xyz (N,3)
        # print(f"Original point num of frame{idx} ", point.shape[0])
        
        for j in range(cam_num):
            cam_depth = np.zeros(load_size, dtype=np.float32) # (640, 960)
            K = Ks[j] # (3, 3)
            fx = float(K[0, 0]) * load_size[1] / ORIGINAL_SIZE[j][1]
            fy = float(K[1, 1]) * load_size[0] / ORIGINAL_SIZE[j][0]
            cx = float(K[0, 2]) * load_size[1] / ORIGINAL_SIZE[j][1]
            cy = float(K[1, 2]) * load_size[0] / ORIGINAL_SIZE[j][0]
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
            point_camera = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ lidar2cam[j].T)[:, :3] # (N,3)
            front_mask = point_camera[:, 2] > 0
            point_camera = point_camera[front_mask]
            print(f"Number of points in front of camera {j}: {np.sum(front_mask)}")
            print(f"Their average depth: {np.mean(point_camera[:,2],axis=0):.5f}")
            print(f"Their max depth: {np.max(point_camera[:,2],axis=0):.5f}")
            print(f"Their min depth: {np.min(point_camera[:,2],axis=0):.5f}")

            
            # project to image plane
            # (3,1) = (3,3) @ (3,1)
            # (1,3) = (1,3) @ (3,3)^T
            point_pixel = point_camera @ K.T
            # normalize 
            point_pixel = point_pixel[:, :2] / point_pixel[:, 2:] 
            
            # check if the points are in the image plane
            mask = np.logical_and.reduce((point_pixel[:, 0] >= 0, point_pixel[:, 0] < load_size[1], 
                                           point_pixel[:, 1] >= 0, point_pixel[:, 1] < load_size[0]))
            
            print(f"Number of points in image plane: {np.sum(mask)}")
            print(f"{np.sum(front_mask)-np.sum(mask)} points are out of image plane")
            
            
            depth_values = point_camera[:, 2] #(N,)

            point_pixel_in = point_pixel[mask]
            depth_in = depth_values[mask]
            u = np.floor(point_pixel_in[:, 0]).astype(np.int32)  # 列坐标，对应 width
            v = np.floor(point_pixel_in[:, 1]).astype(np.int32)  # 行坐标，对应 height
            # 安全裁剪：防止越界
            u = np.clip(u, 0, load_size[1] - 1)  # 宽度方向
            v = np.clip(v, 0, load_size[0] - 1)  # 高度方向

            # 为了避免多个点落在同一个像素，这里我们取最小深度（离相机最近）
            for px_u, px_v, d in zip(u, v, depth_in):
                if cam_depth[px_v, px_u] == 0:  # 如果还没赋值
                    cam_depth[px_v, px_u] = d
                else:
                    cam_depth[px_v, px_u] = min(cam_depth[px_v, px_u], d)

            # 保存或使用 cam_depth，例如保存为 .npy
            save_path = os.path.join(source_path, f"sparse_depth_{j}", car_id + ".npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, cam_depth)
            print(f"Save sparse depth for camera {j} of frame {idx} to {save_path}")
            print(f"Average depth {np.mean(cam_depth[cam_depth > 0]):.2f}")
            print(f"Max depth {np.max(cam_depth[cam_depth > 0]):.2f}")
            print(f"Min depth {np.min(cam_depth[cam_depth > 0]):.2f}")

            rgb_path = os.path.join(source_path, f"image_{j}", car_id + ".png")

            # overlay_depth_on_rgb(depth_path=save_path, rgb_path=rgb_path, vmax=35,save_path=f"/home/DeSiRe-GS/f{idx}_cam{j}_sparse_depth_single")  # Visualize the depth map



def overlay_depth_on_rgb(depth_path, rgb_path, alpha=0.6, vmax=None, save_path=None, hist_save_path=None):
    """
    Overlay sparse depth map on RGB image using jet colormap and save depth histogram.

    Args:
        depth_path (str): Path to .npy depth map.
        rgb_path (str): Path to .png RGB image.
        alpha (float): Transparency factor for overlay.
        vmax (float or None): Max depth for colormap normalization.
        save_path (str or None): Path to save the overlay image, or None to show.
        hist_save_path (str or None): Path to save the depth histogram, or None to skip saving.
    """
    # Load depth and RGB
    depth = np.load(depth_path)  # (H, W)
    print("depth max", depth.max())
    rgb = cv2.imread(rgb_path)   # BGR format
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    assert rgb.shape[:2] == depth.shape, f"RGB image size {rgb.shape[:2]} and depth map size {depth.shape} mismatch."

    # Mask and normalization
    mask = depth > 0
    if vmax is None:
        vmax = np.percentile(depth[mask], 95) if np.any(mask) else 1.0

    norm_depth = np.zeros_like(depth, dtype=np.float32)
    norm_depth[mask] = depth[mask] / vmax
    norm_depth = np.clip(norm_depth, 0, 1)

    # Apply jet colormap
    depth_colored = np.zeros_like(rgb, dtype=np.uint8)
    colormap = plt.get_cmap('jet')
    depth_rgb = (colormap(norm_depth)[:, :, :3] * 255).astype(np.uint8)
    depth_colored[mask] = depth_rgb[mask]

    # Overlay
    overlay = cv2.addWeighted(rgb, 1 - alpha, depth_colored, alpha, 0)

    # Show or save overlay
    plt.figure(figsize=(10, 6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title(f"Overlay Depth on RGB: {os.path.basename(depth_path)}")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved overlay image to {save_path}")
    else:
        plt.show()

    # Save depth histogram
    if hist_save_path and np.any(mask):
        plt.figure(figsize=(8, 4))
        plt.hist(depth[mask].flatten(), bins=100, color='blue', alpha=0.7)
        plt.title(f"Depth Histogram: {os.path.basename(depth_path)}")
        plt.xlabel('Depth')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(hist_save_path, dpi=300)
        print(f"Saved depth histogram to {hist_save_path}")
        plt.close()




if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--source_path", type=str, default = "/home/yangchen/yangchen/datasets/waymo/waymo_data/")
    parser.add_argument("--cam_num", type=int, default=3)
    args = parser.parse_args()
    source_path = args.source_path
    cam_num = args.cam_num
    gt_depth_generator(source_path, cam_num)

    # overlay_depth_on_rgb(depth_path="/data/waymo/pvg_scenes/0158150/dense_depth_2/0158154_pred.npy",\
    #     rgb_path="/data/waymo/pvg_scenes/0158150/image_2/0158154.png",\
    #     alpha=0.6, vmax=60, save_path="/home/DeSiRe-GS/0158154_overlay_est.png",hist_save_path="/home/DeSiRe-GS/0158154_est_hist.png")
    
    # overlay_depth_on_rgb(depth_path="/data/waymo/pvg_scenes/0158150/sparse_depth_2/0158154.npy",\
    #     rgb_path="/data/waymo/pvg_scenes/0158150/image_2/0158154.png",\
    #     alpha=0.6, vmax=60, save_path="/home/DeSiRe-GS/0158154_overlay_gt.png",hist_save_path="/home/DeSiRe-GS/0158154_gt_hist.png")
    



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
import glob
import json
import os
import torch
import torch.nn.functional as F
from utils.loss_utils import psnr, ssim
from gaussian_renderer import get_renderer
from scene import Scene, GaussianModel, EnvLight
from utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf
from pprint import pprint, pformat
from omegaconf import OmegaConf
from texttable import Texttable
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
EPS = 1e-5
non_zero_mean = (
    lambda x: sum(x) / len(x) if len(x) > 0 else -1
)
@torch.no_grad()
def evaluation(iteration, scene : Scene, renderFunc, renderArgs, env_map=None):
    from lpipsPyTorch import lpips

    scale = scene.resolution_scales[0]
    if "kitti" in args.model_path:
        # follow NSG: https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/data_loader/load_kitti.py#L766
        num = len(scene.getTrainCameras())//2
        eval_train_frame = num//5
        traincamera = sorted(scene.getTrainCameras(), key =lambda x: x.colmap_id)
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                            {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:]+traincamera[num:][-eval_train_frame:]})
    else:
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                        {'name': 'train', 'cameras': scene.getTrainCameras()})
    
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = []
            psnr_test = []
            ssim_test = []
            lpips_test = []
            masked_psnr_test = []
            masked_ssim_test = []
            outdir = os.path.join(args.model_path, "eval", "eval_on_" + config['name'] + "data" + "_render")
            image_folder = os.path.join(args.model_path, "images")
            os.makedirs(outdir,exist_ok=True)
            os.makedirs(image_folder,exist_ok=True)
            # opaciity_mask = scene.gaussians.get_opacity[:, 0] > 0.01
            # print("number of valid gaussians: {:d}".format(opaciity_mask.sum().item()))
            for camera_id, viewpoint in enumerate(tqdm(config['cameras'], bar_format="{l_bar}{bar:50}{r_bar}")):
                # render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map, mask=opaciity_mask)
                _, _, render_pkg = renderFunc(args, viewpoint, gaussians, background, scene.time_interval, env_map, iteration, camera_id)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                cv2.imwrite(os.path.join(image_folder, f"{viewpoint.colmap_id:03d}_gt.png"), (gt_image[[2,1,0], :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(image_folder, f"{viewpoint.colmap_id:03d}_render.png"), (image[[2,1,0], :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

                depth = render_pkg['depth']
                alpha = render_pkg['alpha']
                sky_depth = 900
                depth = depth / alpha.clamp_min(EPS)
                if env_map is not None:
                    if args.depth_blend_mode == 0:  # harmonic mean
                        depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                    elif args.depth_blend_mode == 1:
                        depth = alpha * depth + (1 - alpha) * sky_depth
                sky_mask = viewpoint.sky_mask.to("cuda")
                # dynamic_mask = viewpoint.dynamic_mask.to("cuda") if viewpoint.dynamic_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)            
                dynamic_mask = render_pkg['dynamic_mask']
                depth = visualize_depth(depth)
                alpha = alpha.repeat(3, 1, 1)

                grid = [image, alpha, depth, gt_image, torch.logical_not(sky_mask[:1]).float().repeat(3, 1, 1), dynamic_mask.float().repeat(3, 1, 1)]
                grid = make_grid(grid, nrow=3)

                save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))

                l1_test.append(F.l1_loss(image, gt_image).double().item())
                psnr_test.append(psnr(image, gt_image).double().item())
                ssim_test.append(ssim(image, gt_image).double().item())
                lpips_test.append(lpips(image, gt_image, net_type='alex').double().item())  # very slow
                # get the binary dynamic mask
                
                if dynamic_mask.sum() > 0:
                    dynamic_mask = dynamic_mask.repeat(3, 1, 1) > 0 # (C, H, W)
                    masked_psnr_test.append(psnr(image[dynamic_mask], gt_image[dynamic_mask]).double().item())
                    unaveraged_ssim = ssim(image, gt_image, size_average=False) # (C, H, W)
                    masked_ssim_test.append(unaveraged_ssim[dynamic_mask].mean().double().item())
                    

            psnr_test = non_zero_mean(psnr_test)
            l1_test = non_zero_mean(l1_test)
            ssim_test = non_zero_mean(ssim_test)
            lpips_test = non_zero_mean(lpips_test)
            masked_psnr_test = non_zero_mean(masked_psnr_test)
            masked_ssim_test = non_zero_mean(masked_ssim_test)
            
               
            t = Texttable()
            t.add_rows([["PSNR", "SSIM", "LPIPS", "L1", "PSNR (dynamic)", "SSIM (dynamic)"], 
                        [f"{psnr_test:.4f}", f"{ssim_test:.4f}", f"{lpips_test:.4f}", f"{l1_test:.4f}", f"{masked_psnr_test:.4f}", f"{masked_ssim_test:.4f}"]])
            print(t.draw())
            with open(os.path.join(outdir, "metrics.json"), "w") as f:
                json.dump({"split": config['name'], "iteration": iteration,
                    "psnr": psnr_test, "ssim": ssim_test, "lpips": lpips_test, "masked_psnr": masked_psnr_test, "masked_ssim": masked_ssim_test,
                    }, f)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config_path", type=str, required=True)
    params, _ = parser.parse_known_args()
    
    args = OmegaConf.load(params.config_path)
    args.resolution_scales = args.resolution_scales[:1]
    print('Configurations:\n {}'.format(pformat(OmegaConf.to_container(args, resolve=True, throw_on_missing=True))))
    
    seed_everything(args.seed)

    sep_path = os.path.join(args.model_path, 'separation')
    os.makedirs(sep_path, exist_ok=True)
    
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    
    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    print(f"Loading checkpoint {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args)
    
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint), 
                                    os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
        uncertainty_model_path = os.path.join(os.path.dirname(checkpoint), 
                                    os.path.basename(checkpoint).replace("chkpnt", "uncertainty_model"))
        state_dict = torch.load(uncertainty_model_path)
        gaussians.uncertainty_model.load_state_dict(state_dict, strict=False)
            
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_func, render_wrapper = get_renderer(args.render_type)
    evaluation(first_iter, scene, render_wrapper, (args, background), env_map=env_map)

    print("Evaluation complete.")

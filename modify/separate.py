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
import logging
import os
import sys

import numpy as np
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
from pprint import pformat
from scipy.ndimage import binary_dilation
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_renderer import get_renderer
from scene import DynamicGaussianModel, EnvLight, Scene
from utils.general_utils import (
    get_mask_from_projection,
    points_inside_knn_region,
    seed_everything,
    visualize_depth,
)

@torch.no_grad()
def separation(scene: Scene, render_func, render_args, env_map=None, output_path=None):
    """Visualize static/dynamic decomposition for each view."""

    scale = scene.resolution_scales[0]
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras(scale=scale)},
        {"name": "train", "cameras": scene.getTrainCameras()},
    )

    obj_dc_3d = scene.gaussians.get_obj_dc.permute(2, 0, 1)
    logits3d = scene.gaussians.classifier(obj_dc_3d)
    prob = torch.softmax(logits3d, dim=0)
    reliable_mask_all = prob.max(dim=0).values > -1
    id_pred_all = torch.argmax(logits3d, dim=0)
    beta_mask = gaussians.get_scaling_t[:, 0] < args.separate_scaling_t * 1.5
    opacity = gaussians.get_opacity

    for config in validation_configs:
        if not config["cameras"]:
            continue

        outdir = os.path.join(output_path, config["name"])
        os.makedirs(outdir, exist_ok=True)

        for viewpoint in tqdm(config["cameras"]):
            if viewpoint.colmap_id % 5 != 0:
                continue

            valid_id = torch.tensor([0, 1, 2, 3, 4], device="cuda")
            binary_dynamic_mask = viewpoint.dynamic_mask
            binary_dynamic_mask = binary_dynamic_mask.squeeze().cpu().numpy()
            binary_dynamic_mask = binary_dilation(
                binary_dynamic_mask, structure=np.ones((5, 5)), iterations=1
            )
            binary_dynamic_mask = (
                torch.from_numpy(binary_dynamic_mask)
                .to(device="cuda")
                .unsqueeze(0)
            )

            marginal_t = gaussians.get_marginal_t(viewpoint.timestamp)
            exist_mask = (marginal_t[:, 0].detach() > 0.05) & (
                opacity[:, 0].detach() > 0.01
            )
            dynamic_mask = (
                torch.isin(id_pred_all.squeeze(), valid_id)
                & reliable_mask_all.squeeze()
                & beta_mask
            )
            static_mask = ~dynamic_mask & exist_mask
            proj_mask = get_mask_from_projection(
                scene.gaussians._xyz,
                static_mask,
                viewpoint.get_k().float(),
                torch.inverse(viewpoint.c2w),
                binary_dynamic_mask.squeeze(),
                image_width=viewpoint.image_width,
                image_height=viewpoint.image_height,
            )
            static_mask = points_inside_knn_region(
                scene.gaussians._xyz, dynamic_mask, proj_mask, k=5
            )

            render_pkg = render_func(
                viewpoint, scene.gaussians, *render_args, env_map=env_map
            )
            render_pkg_static = render_func(
                viewpoint,
                scene.gaussians,
                *render_args,
                env_map=None,
                mask=static_mask,
            )
            render_pkg_dynamic = render_func(
                viewpoint,
                scene.gaussians,
                *render_args,
                env_map=None,
                mask=dynamic_mask,
            )
            render_pkg_non_dynamic = render_func(
                viewpoint,
                scene.gaussians,
                *render_args,
                env_map=None,
                mask=~dynamic_mask,
            )
            render_pkg_background = render_func(
                viewpoint,
                scene.gaussians,
                *render_args,
                env_map=None,
                mask=~(static_mask | dynamic_mask),
            )

            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            image_static = torch.clamp(render_pkg_static["render"], 0.0, 1.0)
            image_dynamic = torch.clamp(render_pkg_dynamic["render"], 0.0, 1.0)
            image_non_dynamic = torch.clamp(render_pkg_non_dynamic["render"], 0.0, 1.0)
            image_background = torch.clamp(render_pkg_background["render"], 0.0, 1.0)
            alpha_static = render_pkg_background["alpha"].repeat(3, 1, 1)
            depth_static = render_pkg_background["depth"]
            normal_static = render_pkg_background["normal"]

            grid = make_grid(
                [
                    image,
                    image_dynamic,
                    image_static,
                    image_non_dynamic,
                    image_background,
                    alpha_static,
                    visualize_depth(depth_static),
                    normal_static,
                ],
                nrow=8,
                padding=0,
            )
            save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default = "../configs/base.yaml")
    args, _ = parser.parse_known_args()
    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    
    
    args.resolution_scales = args.resolution_scales[:1]
    print('Configurations:\n {}'.format(pformat(OmegaConf.to_container(args, resolve=True, throw_on_missing=True))))


    seed_everything(args.seed)

    dynamic_id_dict = {} 
    # Expected JSON format: {"cam0": [id1, id2], "cam1": [...], ...}
    if args.dynmaic_id_dict_path and os.path.exists(args.dynmaic_id_dict_path):
        with open(args.dynmaic_id_dict_path, 'r') as f:
            dynamic_id_dict = json.load(f)
        logging.info(f"Loaded dynamic_id_dict from {args.dynmaic_id_dict_path}, {len(dynamic_id_dict)} dynamic ids")
    else:
        raise ValueError(f"Dynamic id dict path {args.dynmaic_id_dict_path} does not exist.")
    
    gaussians = DynamicGaussianModel(args)
    scene = Scene(args, gaussians, dynamic_id_dict, shuffle=False)
    
    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    
    for ckpt in [40000,50000,60000]:
        checkpoint = os.path.join(args.model_path, f"chkpnt{ckpt}.pth")     
        output_path = os.path.join(args.model_path, f"separation_{ckpt}_no_beta") 
        print(f"Loading checkpoint {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args)
        
        if env_map is not None:
            env_checkpoint = os.path.join(os.path.dirname(checkpoint), 
                                        os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
            (light_params, _) = torch.load(env_checkpoint)
            env_map.restore(light_params)
        
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_func, _, _, _ = get_renderer(args.render_type)
    
        os.makedirs(output_path,exist_ok=True)
        separation(scene, render_func, (args, background), env_map=env_map,output_path=output_path)





    print("Rendering statics and dynamics complete.")   

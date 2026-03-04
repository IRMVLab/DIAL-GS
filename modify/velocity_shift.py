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

import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
from pprint import pformat
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_renderer import get_renderer
from scene import DynamicGaussianModel, EnvLight, Scene
from utils.general_utils import seed_everything

@torch.no_grad()
def shift_velocity(scene: Scene, render_func, render_args, env_map=None):
    """Shift positions along predicted instantaneous velocity for ID 0."""

    scale = scene.resolution_scales[0]
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras(scale=scale)},
        {"name": "train", "cameras": scene.getTrainCameras()},
    )

    high_mask = scene.gaussians.get_xyz[:, 2] < 0
    obj_dc_3d = scene.gaussians.get_obj_dc.permute(2, 0, 1)
    logits3d = scene.gaussians.classifier(obj_dc_3d)
    prob = torch.softmax(logits3d, dim=0)
    reliable_mask = prob.max(dim=0).values > 0.3
    id_pred = torch.argmax(logits3d, dim=0)
    beta_mask = gaussians.get_scaling_t[:, 0] < args.separate_scaling_t

    for config in validation_configs:
        if not config["cameras"]:
            continue

        outdir = os.path.join(args.model_path, "v_id_0", config["name"])
        os.makedirs(outdir, exist_ok=True)

        for viewpoint in tqdm(config["cameras"]):
            if viewpoint.colmap_id % 5 != 0:
                continue

            original_xyz = scene.gaussians._xyz.clone()
            render_pkg_original = render_func(
                viewpoint, scene.gaussians, *render_args, env_map=None
            )
            image_original = torch.clamp(render_pkg_original["render"], 0.0, 1.0)

            id_mask_0 = (
                (id_pred.squeeze() == 0)
                & reliable_mask.squeeze()
                & (~high_mask)
                & beta_mask
            )

            if id_mask_0.any():
                time_shift = scene.time_interval * 3
                instant_v = scene.gaussians.get_inst_velocity[id_mask_0]
                scene.gaussians._xyz[id_mask_0] += instant_v * time_shift

            render_pkg_modified = render_func(
                viewpoint, scene.gaussians, *render_args, env_map=None
            )
            render_pkg_dynamic = render_func(
                viewpoint,
                scene.gaussians,
                *render_args,
                env_map=None,
                mask=id_mask_0,
            )

            image_modified = torch.clamp(render_pkg_modified["render"], 0.0, 1.0)
            image_dynamic = torch.clamp(render_pkg_dynamic["render"], 0.0, 1.0)

            grid = make_grid(
                [image_original, image_modified, image_dynamic], nrow=3, padding=0
            )
            save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))

            scene.gaussians._xyz = original_xyz



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Velocity shift rendering")
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

    # sep_path = os.path.join(args.model_path, 'separation_beta')
    # os.makedirs(sep_path, exist_ok=True)
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
    shift_velocity(scene, render_func, (args, background), env_map=env_map)


from typing import Optional, Iterable, Sequence, cast, Any, TypeVar
import logging
import urllib.request
import io
import itertools
import random
from tqdm import tqdm
from functools import reduce
from operator import mul
from torch.nn import functional as F
from torch import Tensor
from torch import nn
import types
import math
from pathlib import Path
import os
import timm
import numpy as np
from random import randint
import albumentations as A
import torch
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement

from simple_knn._C import distCUDA2  # type: ignore
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # type: ignore
from scene import dinov2
from scene.fit3d import GetDinov2RegFeats,LoadDinov2Model,get_intermediate_layers

T = TypeVar("T")


def convert_image_dtype(image: np.ndarray, dtype) -> np.ndarray:
    if image.dtype == dtype:
        return image
    if image.dtype != np.uint8 and dtype != np.uint8:
        return image.astype(dtype)
    if image.dtype == np.uint8 and dtype != np.uint8:
        return image.astype(dtype) / 255.0
    if image.dtype != np.uint8 and dtype == np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"cannot convert image from {image.dtype} to {dtype}")


def assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


def camera_project(cameras, xyz: Tensor) -> Tensor:
    eps = torch.finfo(xyz.dtype).eps  # type: ignore
    assert xyz.shape[-1] == 3

    # World -> Camera
    origins = cameras.poses[..., :3, 3]
    rotation = cameras.poses[..., :3, :3]
    # Rotation and translation
    uvw = xyz - origins
    uvw = (rotation * uvw[..., :, None]).sum(-2)

    # Camera -> Camera distorted
    uv = torch.where(uvw[..., 2:] > eps, uvw[..., :2] / uvw[..., 2:], torch.zeros_like(uvw[..., :2]))

    # We assume pinhole camera model in 3DGS anyway
    # uv = _distort(cameras.camera_types, cameras.distortion_parameters, uv, xnp=xnp)

    x, y = torch.moveaxis(uv, -1, 0)

    # Transform to image coordinates
    # Camera distorted -> Image
    fx, fy, cx, cy = torch.moveaxis(cameras.intrinsics, -1, 0)
    x = fx * x + cx
    y = fy * y + cy
    return torch.stack((x, y), -1)


def safe_state():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def scale_grads(values, scale):
    grad_values = values * scale
    rest_values = values.detach() * (1 - scale)
    return grad_values + rest_values


def ssim_down(x, y, max_size=None):
    osize = x.shape[2:]
    if max_size is not None:
        scale_factor = max(max_size/x.shape[-2], max_size/x.shape[-1])
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')
    out = ssim(x, y, size_average=False).unsqueeze(1)
    if max_size is not None:
        out = F.interpolate(out, size=osize, mode='bilinear', align_corners=False)
    return out.squeeze(1)


def _ssim_parts(img1, img2, window_size=11):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma1 = torch.sqrt(sigma1_sq.clamp_min(0))
    sigma2 = torch.sqrt(sigma2_sq.clamp_min(0))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return luminance, contrast, structure


def msssim(x, y, max_size=None, min_size=200):
    raw_orig_size = x.shape[-2:]
    if max_size is not None:
        scale_factor = min(1, max(max_size/x.shape[-2], max_size/x.shape[-1]))
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')

    ssim_maps = list(_ssim_parts(x, y))
    orig_size = x.shape[-2:]
    while x.shape[-2] > min_size and x.shape[-1] > min_size:
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
        ssim_maps.extend(tuple(F.interpolate(x, size=orig_size, mode='bilinear') for x in _ssim_parts(x, y)[1:]))
    out = torch.stack(ssim_maps, -1).prod(-1)
    if max_size is not None:
        out = F.interpolate(out, size=raw_orig_size, mode='bilinear')
    return out.mean(1)


def dino_downsample(x, max_size=None):
    if max_size is None:
        return x
    h, w = x.shape[2:]
    if max_size < h or max_size < w:
        scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
        nh = int(h * scale_factor)
        nw = int(w * scale_factor)
        nh = ((nh + 13) // 14) * 14
        nw = ((nw + 13) // 14) * 14
        x = F.interpolate(x, size=(nh, nw), mode='bilinear')
    return x


class UncertaintyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = getattr(dinov2, config.uncertainty_backbone)(pretrained=True) #getattr表示从dinov2中获取backbone
        self.patch_size = self.backbone.patch_size # 主干网络的patch_size
        in_features = self.backbone.embed_dim # 主干网络输出的特征维度
        # self.conv_seg = nn.Conv2d(in_features, 1, kernel_size=1)
        
        # MLP
        self.conv_seg = nn.Sequential(nn.Conv2d(in_features, 256, kernel_size=3, padding=1), 
                                      nn.ReLU(), 
                                      nn.Conv2d(256, 64, kernel_size=3, padding=1), 
                                      nn.ReLU(),
                                      nn.Conv2d(64, 1, kernel_size=1))
        self.bn = nn.SyncBatchNorm(in_features)
        # nn.init.normal_(self.conv_seg.weight.data, 0, 0.01)
        # nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)

        img_norm_mean = torch.tensor([0.0,  0.0, 0.0], dtype=torch.float32)
        img_norm_std = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer("img_norm_mean", img_norm_mean)
        self.register_buffer("img_norm_std", img_norm_std)
        
        self.max_size = 952 if config.uncertainty_dino_max_size is None else config.uncertainty_dino_max_size

        self._images_cache = {}

        # Freeze dinov2 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._load_model()

    def _load_model(self):
        original_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=False,
            )
        original_model.get_intermediate_layers = types.MethodType(
            get_intermediate_layers,
            original_model
        )
        fine_model = torch.hub.load("FiT3D", "dinov2_reg_small_fine",source='local')
        fine_model.get_intermediate_layers = types.MethodType(
            get_intermediate_layers,
            fine_model
        )
        self.original_model=original_model
        self.fine_model=fine_model
        
    def _get_pad(self, size):
        new_size = math.ceil(size / self.patch_size) * self.patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def _initialize_head_from_checkpoint(self):
        # ADA20 classes to ignore
        cls_to_ignore = [13, 21, 81, 84]
        # Pull the checkpoint
        backbone = self.config.uncertainty_backbone
        url = f"https://dl.fbaipublicfiles.com/dinov2/{backbone}/{backbone}_ade20k_linear_head.pth"
        with urllib.request.urlopen(url) as f:
            checkpoint_data = f.read()
        checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cpu")
        old_weight = checkpoint["state_dict"]["decode_head.conv_seg.weight"] #(N_classes, C_in, 1, 1)
        new_weight = torch.empty(1, old_weight.shape[1], 1, 1)# (1, C_in, 1, 1)
        nn.init.normal_(new_weight, 0, 0.0001)
        new_weight[:, cls_to_ignore] = old_weight[:, cls_to_ignore] * 1000
        nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)
        self.conv_seg.weight.data.copy_(new_weight)

        # Load the bn data
        self.bn.load_state_dict({k[len("decode_head.bn."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("decode_head.bn.")})

    def _get_dino_cached(self, x, cache_entry=None):
        if cache_entry is None or (cache_entry, x.shape) not in self._images_cache:
            with torch.no_grad():
                x = self.backbone.get_intermediate_layers(x, n=[self.backbone.num_heads-1], reshape=True)[-1]
            if cache_entry is not None:
                self._images_cache[(cache_entry, x.shape)] = x.detach().cpu()
        else:
            x = self._images_cache[(cache_entry, x.shape)].to(x.device)
        return x

    def _compute_cosine_similarity(self, x, y, _x_cache=None, _y_cache=None, max_size=None):
        # Normalize data
        h, w = x.shape[2:]
        if max_size is not None and (max_size < h or max_size < w):
            assert max_size % 14 == 0, "max_size must be divisible by 14"
            scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
            nh = int(h * scale_factor)
            nw = int(w * scale_factor)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear')
            y = F.interpolate(y, size=(nh, nw), mode='bilinear')

        x = (x - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        y = (y - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        x = F.pad(x, pads)
        padded_shape = x.shape
        y = F.pad(y, pads)

        with torch.no_grad():
            x = self._get_dino_cached(x, _x_cache)
            y = self._get_dino_cached(y, _y_cache)

        cosine = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        cosine: Tensor = F.interpolate(cosine, size=padded_shape[2:], mode="bilinear", align_corners=False)
        
        # Remove padding
        cosine = cosine[:, :, pads[2]:h+pads[2], pads[0]:w+pads[0]]
        if max_size is not None and (max_size < h or max_size < w):
            cosine = F.interpolate(cosine, size=(h, w), mode='bilinear', align_corners=False)
        return cosine.squeeze(1)
    
    def _forward_uncertainty_features(self, inputs: Tensor, _cache_entry=None) -> Tensor:
        # Normalize data

        transforms = A.Compose([
                    A.Normalize(mean=list(self.img_norm_mean), std=list(self.img_norm_std)),
            ])
        
        x = GetDinov2RegFeats(self.original_model,self.fine_model, inputs, transforms)
     
        x = F.dropout2d(x, p=self.config.uncertainty_dropout, training=self.training)
        x = self.bn(x)
        logits = self.conv_seg(x)
        # We could also do this using weight init, 
        # but we want to have a prior then doing L2 regularization
        logits = logits + math.log(math.exp(1) - 1)

        # Rescale to input size
        logits = F.softplus(logits)
        logits: Tensor = F.interpolate(logits, size=inputs.shape[2:], mode="bilinear", align_corners=False)
        logits = logits.clamp(min=self.config.uncertainty_clip_min)

        # Add padding
        # logits = logits[:, :, pads[2]:h+pads[2], pads[0]:w+pads[0]]
        return logits

    @property
    def device(self):
        return self.img_norm_mean.device

    def forward(self, image: Tensor, _cache_entry=None):
        return self._forward_uncertainty_features(image, _cache_entry=_cache_entry)

    def setup_data(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def _load_image(self, img):
        return torch.from_numpy(np.moveaxis(convert_image_dtype(img, np.float32), -1, 0)[None]).to(self.device)

    def _scale_input(self, x, max_size: Optional[int] = 504):
        h, w = nh, nw = x.shape[2:]
        if max_size is not None:
            scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
            if scale_factor >= 1:
                return x
            nw = int(w * scale_factor)
            nh = int(h * scale_factor)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear')
        return x

    def _dino_plus_ssim(self, gt, prediction, _cache_entry=None, max_size=None):
        gt_down = dino_downsample(gt, max_size=max_size)
        prediction_down = dino_downsample(prediction, max_size=max_size)
        dino_cosine = self._compute_cosine_similarity(
            gt_down,
            prediction_down,
            _x_cache=_cache_entry).detach()
        dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
        msssim_part = 1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
        return torch.min(dino_part, msssim_part)

    def _compute_losses(self, gt, prediction, sky_mask, prefix='', _cache_entry=None):
        uncertainty = self(self._scale_input(gt, None), _cache_entry=_cache_entry)
        log_uncertainty = torch.log(uncertainty)
        # _dssim_go = dssim_go(gt, prediction, size_average=False).unsqueeze(1).clamp_max(self.config.uncertainty_dssim_clip_max)
        # _dssim_go = 1 - ssim(gt, prediction).unsqueeze(1)
        _ssim = ssim_down(gt, prediction, max_size=400).unsqueeze(1)
        _msssim = msssim(gt, prediction, max_size=400, min_size=80).unsqueeze(1)

        if self.config.uncertainty_mode == "l2reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(uncertainty, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = 1 / (2 * uncertainty.pow(2))
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "l1reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(uncertainty, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = 1 / uncertainty
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "dino":
            # loss_mult = 1 / (2 * uncertainty.pow(2))
            # loss_mult = 1 / uncertainty
            
            # Compute dino loss
            loss_mult = 1 / (2 * uncertainty.pow(2))

            gt_down = dino_downsample(gt, max_size=self.max_size)
            prediction_down = dino_downsample(prediction, max_size=self.max_size)

            transforms = A.Compose([
                    A.Normalize(mean=list(self.img_norm_mean), std=list(self.img_norm_std)),
            ])

                    
            gt_feats=GetDinov2RegFeats(self.original_model,self.fine_model,gt_down,transforms)
            rendered_feats=GetDinov2RegFeats(self.original_model,self.fine_model,prediction_down,transforms)

            dino_cosine = F.cosine_similarity(gt_feats, rendered_feats, dim=1).unsqueeze(1)

            dino_cosine: Tensor = F.interpolate(dino_cosine, size=gt_down.shape[2:], mode="bilinear", align_corners=False)

            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            # interpolate the binary sky mask to the original size
            sky_mask = F.interpolate(sky_mask.float(), size=gt_down.shape[2:], mode='bilinear', align_corners=False).bool()
            dino_part = dino_part.masked_fill(sky_mask, 0.0)
            uncertainty_loss = dino_part * dino_downsample(loss_mult, max_size=self.max_size)
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(loss_mult, size=gt.shape[2:], mode='bilinear', align_corners=False)
            if dino_part.shape[2:] != gt.shape[2:]:
                dino_part = F.interpolate(dino_part, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = loss_mult.clamp_max(3)

        elif self.config.uncertainty_mode == "dino+mssim":
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=self.max_size)
            prediction_down = dino_downsample(prediction, max_size=self.max_size)
            dino_cosine = self._compute_cosine_similarity(
                gt_down,
                prediction_down,
                _x_cache=_cache_entry).detach()
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            msssim_part = 1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
            uncertainty_loss = torch.min(dino_part, msssim_part) * dino_downsample(loss_mult, max_size=self.max_size)
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(loss_mult, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = loss_mult.clamp_max(3)

        else:
            raise ValueError(f"Invalid uncertainty_mode: {self.config.uncertainty_mode}")

        beta = log_uncertainty.mean()
        loss = uncertainty_loss.mean() + self.config.uncertainty_regularizer_weight * beta

        ssim_discounted = (_ssim * loss_mult).sum() / loss_mult.sum()
        mse = torch.pow(gt - prediction, 2)
        mse_discounted = (mse * loss_mult).sum() / loss_mult.sum()
        psnr_discounted = 10 * torch.log10(1 / mse_discounted)

        metrics = {
            f"{prefix}loss": loss.item(),
            f"{prefix}ssim": _ssim.mean().item(),
            f"{prefix}msssim": _msssim.mean().item(),
            f"{prefix}ssim_discounted": ssim_discounted.item(),
            f"{prefix}mse_discounted": mse_discounted.item(),
            f"{prefix}psnr_discounted": psnr_discounted.item(),
            f"{prefix}beta": beta.item(),
        }
        return loss, metrics, loss_mult.detach(), dino_part.detach()

    def get_loss(self, gt_image, image, sky_mask, prefix='', _cache_entry=None):
        gt_torch = gt_image.unsqueeze(0)
        image = image.unsqueeze(0)
        sky_mask = sky_mask.unsqueeze(0)
        loss, metrics, loss_mult, dino_part = self._compute_losses(gt_torch, image, sky_mask, prefix, _cache_entry=_cache_entry)
        loss_mult = loss_mult.squeeze(0)
        dino_part = dino_part.squeeze(0)
        metrics[f"{prefix}uncertainty_loss"] = metrics.pop(f"{prefix}loss")
        metrics.pop(f"{prefix}ssim")
        return loss, metrics, loss_mult, dino_part

    @staticmethod
    def load(path: str, config: OmegaConf) -> "UncertaintyModel":
        ckpt = torch.load(os.path.join(path), map_location="cpu")

        model = UncertaintyModel(config)
        model.load_state_dict(ckpt, strict=False)
        return model

    def save(self, path: str):
        state = self.state_dict()
        state["config"] = OmegaConf.to_yaml(self.config, resolve=True)
        torch.save(state, os.path.join(path))


#
# SH eval
#
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def getWorld2View2(R, t, translate, scale):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
    z_sign = 1.0
    P = torch.zeros((4, 4))
    P[0, 0] = 2.0 * fx / w
    P[1, 1] = 2.0 * fy / h
    # P[0, 2] = (w - 2.0 * cx) / w
    P[0, 2] = (2.0 * cx - w) / w
    P[1, 2] = (2.0 * cy - h) / h
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def build_rotation(r, device):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


# SSIM
def ssim(img1, img2, window_size=11, size_average=True):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-3)


# SSIM
def dssim_go(img1, img2, window_size=11, size_average=True):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma1 = torch.sqrt(sigma1_sq.clamp_min(0))
    sigma2 = torch.sqrt(sigma2_sq.clamp_min(0))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    # Normal dssim would use this
    # dssim_map = (1 - luminance * contrast * structure) / 2
    # NeRF on the Go uses this:
    dssim_map = (1 - luminance) * (1 - contrast) * (1 - structure)

    if size_average:
        return dssim_map.mean()
    else:
        return dssim_map.mean(-3)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def get_uniform_points_on_sphere_fibonacci(num_points, *, dtype=None, xnp=torch):
    # https://arxiv.org/pdf/0912.4540.pdf
    # Golden angle in radians
    if dtype is None:
        dtype = xnp.float32
    phi = math.pi * (3. - math.sqrt(5.))
    N = (num_points - 1) / 2
    i = xnp.linspace(-N, N, num_points, dtype=dtype)
    lat = xnp.arcsin(2.0 * i / (2*N+1))
    lon = phi * i

    # Spherical to cartesian
    x = xnp.cos(lon) * xnp.cos(lat)
    y = xnp.sin(lon) * xnp.cos(lat)
    z = xnp.sin(lat)
    return xnp.stack([x, y, z], -1)


@torch.no_grad()
def get_sky_points(num_points, points3D: Tensor, cameras):
    xnp = torch
    points = get_uniform_points_on_sphere_fibonacci(num_points, xnp=xnp)
    points = points.to(points3D.device)
    mean = points3D.mean(0)[None]
    sky_distance = xnp.quantile(xnp.linalg.norm(points3D - mean, 2, -1), 0.97) * 10
    points = points * sky_distance
    points = points + mean
    gmask = torch.zeros((points.shape[0],), dtype=xnp.bool, device=points.device)
    for cam in tqdm(cameras, desc="Generating skybox"):
        uv = camera_project(cam, points[xnp.logical_not(gmask)])
        mask = xnp.logical_not(xnp.isnan(uv).any(-1))
        # Only top 2/3 of the image
        assert cam.image_sizes is not None
        mask = xnp.logical_and(mask, uv[..., -1] < 2/3 * cam.image_sizes[..., 1])
        gmask[xnp.logical_not(gmask)] = xnp.logical_or(gmask[xnp.logical_not(gmask)], mask)
    return points[gmask], sky_distance / 2


def add_fourier_features(features: torch.Tensor, scale=(0.0, 1.0), num_frequencies=3):
    features = (features - scale[0]) / (scale[1] - scale[0])
    freqs = 2**torch.linspace(0, num_frequencies-1, num_frequencies, dtype=features.dtype, device=features.device)
    offsets = torch.tensor([0, 0.5 * math.pi], dtype=features.dtype, device=features.device)
    sin_cos_features = torch.sin((2*math.pi * (freqs[..., None, :] * features[..., None]).view(*freqs.shape[:-1], -1)).unsqueeze(-1).add(offsets)).view(*features.shape[:-1], -1)
    return torch.cat((features, sin_cos_features), -1)


def srgb_to_linear(img):
    limit = 0.04045

    # NOTE: torch.where is not differentiable, so we use the following
    # return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

    mask = img > limit
    out = img / 12.92
    out[mask] = torch.pow((img[mask] + 0.055) / 1.055, 2.4)
    return out


def linear_to_srgb(img):
    limit = 0.0031308

    # NOTE: torch.where is not differentiable, so we use the following
    # return torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

    mask = img > limit
    out = 12.92 * img
    out[mask] = 1.055 * torch.pow(img[mask], 1.0 / 2.4) - 0.055
    return out


def get_cameras_extent(cameras):
    c2w = cameras.poses
    cam_centers = c2w[:, :3, 3:4]
    cam_centers = np.hstack(list(cam_centers))
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    # center = center.flatten()
    radius = diagonal * 1.1
    # translate = -center
    return radius


class ResidualBlock(nn.Sequential):
    def forward(self, input):
        x = super().forward(input)
        minch = min(x.size(1), input.size(1))
        return input[:, :minch] + x[:, :minch]
    

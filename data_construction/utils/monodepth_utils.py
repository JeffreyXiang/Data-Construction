from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision.transforms import Compose

from ..models.depth_anything.dpt import DepthAnything
from ..models.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


__all__ = [
    'load_zoedepth_model',
    'load_depthanything_model',
    'zoedepth_predict_depth',
    'depthanything_predict_disparity',
]


def load_zoedepth_model(model_path="isl-org/ZoeDepth", device='cuda'):
    model_zoe_nk = torch.hub.load(model_path, "ZoeD_NK", pretrained=False)
    pretrained_dict = torch.hub.load_state_dict_from_url('https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt', map_location='cpu')
    model_zoe_nk.load_state_dict(pretrained_dict['model'], strict=False)
    for b in model_zoe_nk.core.core.pretrained.model.blocks:
        b.drop_path = nn.Identity()
    return model_zoe_nk.to(device).eval()


def load_depthanything_model(model_path="LiheYoung/depth_anything_vitl14", device='cuda'):
    depth_anything = DepthAnything.from_pretrained(model_path).eval().to(device)
    return depth_anything


@torch.no_grad()
def zoedepth_predict_depth(zeo_depth_model, image: torch.Tensor):
    """
    Predict depth from image using ZoeDepth.

    Args:
        image: (N, 3, H, W) tensor of images.
    Returns:
        depth: (N, H, W) tensor of linearized depth.
    """
    depth_tensor = zeo_depth_model.infer(image).squeeze(1)
    return depth_tensor


@torch.no_grad()
def depthanything_predict_disparity(depth_anything, image: torch.Tensor):
    """
    Predict disparity from image using DepthAnything.

    Args:
        image: (N, 3, H, W) tensor of images.
    Returns:
        depth: (N, H, W) tensor of linearized depth.
    """
    H, W = image.shape[2:]
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    device = image.device

    image = image.permute(0, 2, 3, 1).cpu().numpy()
    image = [transform({'image': image_i})['image'] for image_i in image]
    image = np.stack(image, axis=0)
    image = torch.from_numpy(image).float().to(device)

    depth_tensor = depth_anything(image)
    depth_tensor = F.interpolate(depth_tensor.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
    return depth_tensor

import os
import sys
import json
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import numpy as np

from .diffusion import backbones
from .diffusion import frameworks
from .diffusion import samplers


def load(model_name='sdipdogs_depth_aware_inpaint', device='cuda'):
    with open(os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.json'), 'r') as f:
        cfg = edict(json.load(f))
    backbone = getattr(backbones, cfg.backbone.name)(**cfg.backbone.args)
    framework = getattr(frameworks, cfg.framework.name)(backbone, **cfg.framework.args)
    ## Load checkpoint
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.pt'), map_location='cpu')
    backbone.load_state_dict(ckpt)
    backbone.eval()
    backbone.to(device)
    sampler = samplers.DdimSampler(framework)
    return sampler


@torch.no_grad()
def inpaint(model, image: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor):
    """
    Depth-aware inpainting using ivid.

    Args:
        image: (N, 3, H, W) tensor of images.
        depth: (N, H, W) tensor of depths.
        mask: (N, H, W) tensor of masks.
    Returns:
        image: (N, 3, H, W) tensor of inpainted images.
    """
    N, C, H, W = image.shape
    image = image.cuda()
    depth = depth.cuda()
    mask = mask.cuda()

    # Depth to z-buffer
    depth = torch.where(mask, depth, 1e10 * torch.ones_like(depth))
    near = depth.reshape(N, -1).min(dim=1)[0].reshape(N, 1, 1)
    near = torch.minimum(near, 1.0 * torch.ones_like(near))
    far = 5
    depth = (1 / near - 1 / depth) / (1 / near - 1 / far)
    depth = torch.clamp(depth, 0, 1)

    # Normalize image
    image = image * 2 - 1
    depth = depth.unsqueeze(1) * 2 - 1
    mask = mask.unsqueeze(1)

    # Erode mask
    mask_rgb = F.max_pool2d(~mask*1.0, 3, stride=1, padding=1) < 0.5
    
    args = {
        'y': torch.cat([image, depth], dim=1).float(),
        'mask': mask.float(),
        'mask_rgb': mask_rgb.float(),
        # 'replace_rgb': (
        #     0.1,
        #     image.float(),
        #     mask_rgb.float()
        # ),
        # 'replace_depth': (
        #     0.2,
        #     depth.float(),
        #     mask.float()
        # ),
        # 'constrain_depth': (
        #     0.5,
        #     depth.float(),
        # ),
    }
    res = model.sample(N, steps=50, **args, verbose=False)
    image = res.samples[:, :3] * 0.5 + 0.5
    
    if torch.isnan(image).any():
        raise ValueError("NaN detected in inpainted image")

    return image

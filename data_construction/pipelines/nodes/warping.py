import math

import numpy as np
import torch
import torch.nn.functional as F
from typing import *
import utils3d

from .base import Node


class FovSetting(Node):
    def __init__(self, out_prefix: str = "", fov: float = 45):
        super().__init__("", out_prefix)
        self.fov = fov
    
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Set field of view.
        """
        N = data['image'].shape[0]
        device = data['image'].device

        data[f'{self.out_prefix}fov'] = torch.tensor([self.fov] * N, dtype=torch.float32).to(device)
        return data


class FovRandomSetting(Node):
    def __init__(self, out_prefix: str = "", fov_range: List[float] = [20, 70]):
        super().__init__("", out_prefix)
        self.fov_range = fov_range
    
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Set random field of view.
        """
        data[f'{self.out_prefix}fov'] = torch.rand(data['image'].shape[0], dtype=torch.float32) * (self.fov_range[1] - self.fov_range[0]) + self.fov_range[0]
        return data

    
class Warping(Node):
    def __init__(self, 
        in_image_key: str = "image", in_depth_key: str = "depth", 
        out_image_key: str = "warped_image", out_depth_key: str = "warped_depth", out_mask_key: str = "warped_mask", 
        out_uv_key: str = "warped_uv", out_dr_key: str = "warped_dr", 
        ctx: Literal['gl', 'cuda'] = 'cuda', 
        transform: torch.Tensor = None, 
        mask_cfg_override: Dict[str, float] = {}
    ):
        super().__init__()
        
        DEFAULT_MASK_CFG = {
            'stretching_thresh': 2,
            'depth_diff_thresh': {
                'atol': 0.2,
                'rtol': 0.2,
            },
            'erosion_radius': 0,
        }
        self.in_image_key = in_image_key
        self.in_depth_key = in_depth_key
        self.out_image_key = out_image_key
        self.out_depth_key = out_depth_key
        self.out_mask_key = out_mask_key
        self.out_uv_key = out_uv_key
        self.out_dr_key = out_dr_key

        self.ctx = ctx
        self.transform = transform
        self.mask_cfg = DEFAULT_MASK_CFG
        self.mask_cfg.update(mask_cfg_override)

    @staticmethod
    def _get_depth_diff_mask(depth, atol, rtol):
        def depth_diff(x, y):
            x = torch.clamp(x, min=1e-6)
            y = torch.clamp(y, min=1e-6)
            diff = torch.abs(x - y)
            inv_diff = torch.abs(1 / x - 1 / y)
            return torch.logical_and(diff > atol, inv_diff > rtol)
        mask = torch.zeros(depth.shape, dtype=torch.uint8, device=depth.device)
        mask_ = depth_diff(depth[:, :, 1:], depth[:, :, :-1])
        mask[:, :, 1:] += mask_
        mask[:, :, :-1] += mask_
        mask_ = depth_diff(depth[:, 1:, :], depth[:, :-1, :])
        mask[:, 1:, :] += mask_
        mask[:, :-1, :] += mask_
        mask_ = depth_diff(depth[:, 1:, 1:], depth[:, :-1, :-1])
        mask[:, 1:, 1:] += mask_
        mask[:, :-1, :-1] += mask_
        mask_ = depth_diff(depth[:, 1:, :-1], depth[:, :-1, 1:])
        mask[:, 1:, :-1] += mask_
        mask[:, :-1, 1:] += mask_
        return mask < 3

    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Warp image using depth.

        Args:
            image: (N, 3, H, W) tensor of images.
            depth: (N, H, W) tensor of linearized depth.
            fov: (N) tensor of field of view.
        Returns:
            warped_image: (N, 3, H, W) tensor of warped images.
            warped_depth: (N, H, W) tensor of warped depth.
            warped_mask: (N, H, W) tensor of warped mask.
            warped_uv: (N, 2, H, W) tensor of warped uv.
            warped_dr: (N, 6, H, W) tensor of warping image-space derivatives (du/dx, du/dy, dv/dx, dv/dy, l', s').
        """
        N, _, H, W = data[self.in_image_key].shape
        device = data[self.in_image_key].device
        if not isinstance(self.ctx, str):
            rast_ctx = self.ctx
        else:
            rast_ctx = self.get_lazy_component('rast_ctx', utils3d.torch.RastContext, init_kwargs={'backend': self.ctx}, pipe=pipe)
            self.ctx = rast_ctx

        # Warping
        intrinsics = utils3d.torch.intrinsics_from_fov(torch.deg2rad(data['fov']), H, W, True).to(device)
        extrinsics_src = torch.eye(4, dtype=torch.float32).to(device).unsqueeze(0).repeat(N, 1, 1)
        extrinsics_tgt = self.transform
        warped_image, warped_depth, warped_mask, warped_uv, warped_dr = utils3d.torch.warp_image_by_depth(
            rast_ctx,
            data[self.in_depth_key],
            data[self.in_image_key],
            extrinsics_src=extrinsics_src,
            extrinsics_tgt=extrinsics_tgt,
            intrinsics_src=intrinsics,
            intrinsics_tgt=intrinsics,
            return_uv=True,
            return_dr=True,
            padding=2 * max(H, W),
        )

        data[self.out_image_key] = warped_image
        data[self.out_depth_key] = warped_depth
        data[self.out_uv_key] = warped_uv

        # Use the smaller singular value of 2x2 deformation matrix [[du/dx, du/dy], [dv/dx, dv/dy]] to meature the stretch (while the larger singular value means compression)
        dudx, dudy, dvdx, dvdy = (warped_dr * torch.tensor([W, H, W, H]).to(warped_dr)).unbind(dim=-1)
        s1 = dudx ** 2 + dudy ** 2 + dvdx ** 2 + dvdy ** 2
        s2 = ((dudx ** 2 + dudy ** 2 - dvdx ** 2 - dvdy ** 2) ** 2 + 4 * (dudx * dvdx + dudy * dvdy) ** 2) ** 0.5
        l_prime = (0.5 * (s1 - s2).abs()) ** 0.5        # least singular value
        s_prime = dudx * dvdy - dudy * dvdx             # determinant 

        data[self.out_dr_key] = torch.stack([dudx, dudy, dvdx, dvdy, l_prime, s_prime], dim=1).reshape(N, 6, H, W)
        
        # Masking
        ## stretching caused by warping
        if self.mask_cfg['stretching_thresh'] is not None:
            mask = warped_mask * (l_prime > 1 / self.mask_cfg['stretching_thresh']) * (s_prime < 0)
        ## depth difference
        if self.mask_cfg['depth_diff_thresh'] is not None:
            mask *= self._get_depth_diff_mask(warped_depth, **self.mask_cfg['depth_diff_thresh'])
        ## erosion
        if self.mask_cfg['erosion_radius'] is not None and self.mask_cfg['erosion_radius'] > 0:
            mask = F.max_pool2d(~mask.unsqueeze(1).float(), 2 * self.mask_cfg['erosion_radius'] + 1, stride=1, padding=self.mask_cfg['erosion_radius']).squeeze(1) < 0.5
        data[self.out_mask_key] = mask

        return data
    

class RandomWarping(Warping, Node):
    def __init__(self,  
        in_image_key: str = "image", in_depth_key: str = "depth", 
        out_image_key: str = "warped_image", out_depth_key: str = "warped_depth", out_mask_key: str = "warped_mask", 
        out_uv_key: str = "warped_uv", out_dr_key: str = "warped_dr", 
        ctx: Literal['gl', 'cuda'] = 'cuda',  
        noise_override: Dict[str, float] = {}, mask_cfg_override: Dict[str, float] = {}
    ):
        super().__init__(in_prefix, out_prefix, ctx, None, mask_cfg_override)
        DEFAULT_NOISE = {
            'uv': 0.10,
            'depth': 0.10,
            'radius': 0.10,
            'yaw': 30,
            'pitch': 10,
            'roll': 5,
        }
        self.noise_settings = DEFAULT_NOISE.copy()
        self.noise_settings.update(noise_override)
    
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Warp image using depth.

        Args:
            image: (N, 3, H, W) tensor of images.
            depth: (N, H, W) tensor of linearized depth.
            point: (N, 2) tensor of points.
            fov: (N) tensor of field of view.
        Returns:
            warped_image: (N, 3, H, W) tensor of warped images.
            warped_depth: (N, H, W) tensor of warped depth.
            warped_mask: (N, H, W) tensor of warped mask.
            warped_uv: (N, 2, H, W) tensor of warped uv.
            warped_dr: (N, 6, H, W) tensor of warping image-space derivatives (du/dx, du/dy, dv/dx, dv/dy, l', s').
        """
        N, _, H, W = data[f'{self.in_prefix}image'].shape
        device = data[f'{self.in_prefix}image'].device

        intrinsics = utils3d.torch.intrinsics_from_fov(torch.deg2rad(data['fov']), H, W, True).to(device)   # (N, 3, 3)
        extrinsics_src = torch.eye(4, dtype=torch.float32).to(device).unsqueeze(0).repeat(N, 1, 1)          # (N, 4, 4)

        center_uv = torch.randn(N, 1, 2).to(device) * self.noise_settings['uv'] + 0.5                       # (N, 1, 2)
        if 'point' in data:
            center_depth = data['depth'][
                torch.arange(N),
                torch.floor(data['point'][..., 1] * H).long(),
                torch.floor(data['point'][..., 0] * W).long(),
            ][..., None]  # (N, 1)
        else:
            center_depth = torch.ones(N, 1).to(data['depth'])
        center_depth *= torch.randn(N, 1).to(device) * self.noise_settings['depth'] + 1.0                   # (N, 1)
        center = utils3d.torch.unproject_cv(center_uv, center_depth, extrinsics_src, intrinsics).squeeze(-2)# (N, 3)
        yaw = torch.randn(N).to(device) * math.radians(self.noise_settings['yaw'])
        pitch = torch.randn(N).to(device) * math.radians(self.noise_settings['pitch']) 
        roll = torch.randn(N).to(device) * math.radians(self.noise_settings['roll'])
        R = utils3d.torch.euler_angles_to_matrix(torch.stack([pitch, yaw, roll], dim=-1), 'ZXY')            # (N, 3, 3)
        T = center + (R @ -center.unsqueeze(-1)).squeeze(-1) * (torch.randn(N).to(device) * self.noise_settings['radius'] + 1.0).unsqueeze(-1)  # (N, 3)
        extrinsics_tgt = torch.eye(4, dtype=torch.float32).to(device).unsqueeze(0).repeat(N, 1, 1)          # (N, 4, 4)
        extrinsics_tgt[:, :3, :3] = R
        extrinsics_tgt[:, :3, 3] = T
        data['transform'] = extrinsics_tgt

        self.transform = extrinsics_tgt
        data = super().__call__(data, pipe=pipe)
        return data


class RandomWarping2(Warping, Node):
    """
    A random warping strategy for full scene. It ensures that camera does not get into walls or other objects.

    1. choose a random target camera position by "step-into"/"step-out"/"near-by" strategy
    2. random choose a point near the center of the image and roughly make sure that its gaze is not changed by the transformation
    3. additionally rotate the camera orientation by random (yaw, pitch, roll) angles (proportional to FOV).
    """
    def __init__(self, 
        in_image_key: str = "image", in_depth_key: str = "depth", 
        out_image_key: str = "warped_image", out_depth_key: str = "warped_depth", out_mask_key: str = "warped_mask", 
        out_uv_key: str = "warped_uv", out_dr_key: str = "warped_dr", 
        ctx: Literal['gl', 'cuda'] = 'cuda', 
        noise_override: Dict[str, float] = {}, mask_cfg_override: Dict[str, float] = {}
    ):
        super().__init__(in_image_key, in_depth_key, out_image_key, out_depth_key, out_mask_key, out_uv_key, out_dr_key, ctx, None, mask_cfg_override)
        DEFAULT_NOISE = {
            'x': 0.10,
            'y': 0.10,
            'z': 0.10,
            'depth': 0.10,
            'radius': 0.10,
            'yaw': 30,
            'pitch': 10,
            'roll': 5,
        }
        self.noise_settings = DEFAULT_NOISE.copy()
        self.noise_settings.update(noise_override)
    
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Warp image using depth.

        Args:
            image: (N, 3, H, W) tensor of images.
            depth: (N, H, W) tensor of linearized depth.
            point: (N, 2) tensor of points.
            fov: (N) tensor of field of view.
        Returns:
            warped_image: (N, 3, H, W) tensor of warped images.
            warped_depth: (N, H, W) tensor of warped depth.
            warped_mask: (N, H, W) tensor of warped mask.
            warped_uv: (N, 2, H, W) tensor of warped uv.
            warped_dr: (N, 6, H, W) tensor of warping image-space derivatives (du/dx, du/dy, dv/dx, dv/dy, l', s').
        """
        N, _, H, W = data[f'{self.in_prefix}image'].shape
        device = data[f'{self.in_prefix}image'].device

        intrinsics = utils3d.torch.intrinsics_from_fov(torch.deg2rad(data['fov']), H, W, True).to(device)   # (N, 3, 3)
        extrinsics_src = torch.eye(4, dtype=torch.float32).to(device).unsqueeze(0).repeat(N, 1, 1)          # (N, 4, 4)

        center_uv = torch.randn(N, 1, 2).to(device) * self.noise_settings['uv'] + 0.5                       # (N, 1, 2)
        if 'point' in data:
            center_depth = data['depth'][
                torch.arange(N),
                torch.floor(data['point'][..., 1] * H).long(),
                torch.floor(data['point'][..., 0] * W).long(),
            ][..., None]  # (N, 1)
        else:
            center_depth = torch.ones(N, 1).to(data['depth'])
        center_depth *= torch.randn(N, 1).to(device) * self.noise_settings['depth'] + 1.0                   # (N, 1)
        center = utils3d.torch.unproject_cv(center_uv, center_depth, extrinsics_src, intrinsics).squeeze(-2)# (N, 3)
        yaw = torch.randn(N).to(device) * math.radians(self.noise_settings['yaw'])
        pitch = torch.randn(N).to(device) * math.radians(self.noise_settings['pitch']) 
        roll = torch.randn(N).to(device) * math.radians(self.noise_settings['roll'])
        R = utils3d.torch.euler_angles_to_matrix(torch.stack([pitch, yaw, roll], dim=-1), 'ZXY')            # (N, 3, 3)
        T = center + (R @ -center.unsqueeze(-1)).squeeze(-1) * (torch.randn(N).to(device) * self.noise_settings['radius'] + 1.0).unsqueeze(-1)  # (N, 3)
        extrinsics_tgt = torch.eye(4, dtype=torch.float32).to(device).unsqueeze(0).repeat(N, 1, 1)          # (N, 4, 4)
        extrinsics_tgt[:, :3, :3] = R
        extrinsics_tgt[:, :3, 3] = T
        data['transform'] = extrinsics_tgt

        self.transform = extrinsics_tgt
        data = super().__call__(data, pipe=pipe)
        return data


class BackWarping(Warping, Node):
    def __init__(self, in_prefix: List[str] = ["inpainted_warped_", "warped_"], out_prefix: str = "backwarped_+", ctx='cuda', mask_cfg_override: Dict[str, float] = {}):
        super().__init__(in_prefix[0], out_prefix, ctx, None, mask_cfg_override)
        self.uv_prefix = in_prefix[1]
    
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Backwarping.

        Args:
            image: (N, 3, H, W) tensor of images.
            depth: (N, H, W) tensor of depth.
            mask: (N, H, W) tensor of mask.
            uv: (N, 2, H, W,) tensor of uv.
            transform: (N, 4, 4) tensor of transform.
            fov: (N) tensor of field of view.
        Returns:
            backwarped_image: (N, 3, H, W) tensor of backwarped images.
            backwarped_depth: (N, H, W) tensor of backwarped depth.
            backwarped_mask: (N, H, W) tensor of backwarped mask.
            backwarped_uv: (N, 2, H, W) tensor of backwarped uv.
            backwarped_dr: (N, 6, H, W) tensor of backwarping image-space derivatives (du/dx, du/dy, dv/dx, dv/dy, l', s').
            backwarped_flow: (N, 3, H, W) tensor of flow between src and backwarped.
        """
        transform = data['transform']
        self.transform = torch.inverse(transform)
        data = super().__call__(data, pipe=pipe)

        # Cascade uv
        N, _, H, W = data[f'{self.in_prefix}image'].shape
        device = data[f'{self.in_prefix}image'].device
        uv = data[f'{self.uv_prefix}uv']
        mask = data[f'{self.uv_prefix}mask']
        uvm = torch.cat([uv, mask.float().unsqueeze(1)], dim=1)
        uvm = F.grid_sample(uvm, data[f'{self.out_prefix}uv'].permute(0, 2, 3, 1) * 2 - 1, mode='bilinear', padding_mode='zeros', align_corners=False)
        data[f'{self.out_prefix}flow'] = uvm
        data[f'{self.out_prefix}flow'][:, :2] -= utils3d.torch.image_uv(H, W).float().to(device).permute(2, 0, 1).unsqueeze(0)
        return data
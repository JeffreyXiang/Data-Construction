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
    def __init__(self, in_prefix: str = "", out_prefix: str = "warped_+", ctx='cuda', transform: torch.Tensor = None, mask_cfg_override: Dict[str, float] = {}):
        super().__init__(in_prefix, out_prefix)
        
        DEFAULT_MASK_CFG = {
            'stretching_thresh': 2,
            'depth_diff_thresh': {
                'atol': 0.2,
                'rtol': 0.2,
            },
            'erosion_radius': 0,
        }
        
        self.ctx = ctx
        self.transform = transform
        self.mask_cfg = DEFAULT_MASK_CFG.copy()
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
        N, _, H, W = data['image'].shape
        device = data['image'].device
        if isinstance(self.ctx, utils3d.torch.RastContext):
            rast_ctx = self.ctx
        else:
            rast_ctx = self.get_lazy_component('rast_ctx', utils3d.torch.RastContext, init_kwargs={'backend': self.ctx}, pipe=pipe)

        # Warping
        intrinsics = utils3d.torch.intrinsics_from_fov(torch.deg2rad(data['fov']), H, W, True).to(device)
        extrinsics_src = torch.eye(4, dtype=torch.float32).to(device).unsqueeze(0).repeat(N, 1, 1)
        extrinsics_tgt = self.transform
        warped = utils3d.torch.warp_image_by_depth(
            rast_ctx,
            data[f'{self.in_prefix}depth'],
            data[f'{self.in_prefix}image'],
            extrinsics_src=extrinsics_src,
            extrinsics_tgt=extrinsics_tgt,
            intrinsics_src=intrinsics,
            intrinsics_tgt=intrinsics,
            return_uv=True,
            return_dr=True,
            padding=2 * max(H, W),
        )

        data[f'{self.out_prefix}image'] = warped[0]
        data[f'{self.out_prefix}depth'] = warped[1]
        data[f'{self.out_prefix}uv'] = warped[3]

        # Calculate Derivatives
        dudx = warped[4][:, 0] * W
        dudy = warped[4][:, 1] * H
        dvdx = warped[4][:, 2] * W
        dvdy = warped[4][:, 3] * H
        ## least singular value of the Jacobian of the warp
        l_prime = torch.sqrt(torch.clamp(
            dudx**2 + dudy**2 + dvdx**2 + dvdy**2 -
            torch.sqrt(torch.clamp(
                (dudx**2 + dudy**2 - dvdx**2 - dvdy**2)**2 +
                4 * (dudx * dvdx + dudy * dvdy)**2,
                min=0.0)
            ), min=0.0)
        ) * np.sqrt(0.5)
        ## determinant of the Jacobian of the warp
        s_prime = dudx * dvdy - dudy * dvdx
        data[f'{self.out_prefix}dr'] = torch.stack([dudx, dudy, dvdx, dvdy, l_prime, s_prime], dim=1).reshape(N, 6, H, W)
        
        # Masking
        ## stretching caused by warping
        if self.mask_cfg['stretching_thresh'] is not None:
            mask = warped[2] * (l_prime > 1 / self.mask_cfg['stretching_thresh']) * (s_prime < 0)
        ## depth difference
        if self.mask_cfg['depth_diff_thresh'] is not None:
            mask *= self._get_depth_diff_mask(warped[1], **self.mask_cfg['depth_diff_thresh'])
        ## erosion
        if self.mask_cfg['erosion_radius'] is not None and self.mask_cfg['erosion_radius'] > 0:
            mask = F.max_pool2d(~mask.unsqueeze(1)*1.0, 2 * self.mask_cfg['erosion_radius'] + 1, stride=1, padding=self.mask_cfg['erosion_radius']).squeeze(1) < 0.5
        data[f'{self.out_prefix}mask'] = mask

        return data
    

class RandomWarping(Warping, Node):
    def __init__(self, in_prefix: str = "", out_prefix: str = "warped_+", ctx='cuda', noise_override: Dict[str, float] = {}, mask_cfg_override: Dict[str, float] = {}):
        super().__init__(in_prefix, out_prefix, ctx, None, mask_cfg_override)
        DEFAULT_NOISE = {
            'uv': 0.10,
            'depth': 0.10,
            'radius': 0.10,
            'yaw': 0.30,
            'pitch': 0.10,
            'roll': 0.05,
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
        N, _, H, W = data['image'].shape
        device = data['image'].device

        intrinsics = utils3d.torch.intrinsics_from_fov(torch.deg2rad(data['fov']), H, W, True).to(device) # (N, 3, 3)
        extrinsics_src = torch.eye(4, dtype=torch.float32).to(device).unsqueeze(0).repeat(N, 1, 1)  # (N, 4, 4)
        uv = torch.randn(N, 1, 2).to(device) * self.noise_settings['uv'] + 0.5  # (N, 1, 2)
        depth = data['depth'][
            torch.arange(N),
            torch.floor(data['point'][..., 1] * H).long(),
            torch.floor(data['point'][..., 0] * W).long(),
        ][..., None]  # (N, 1)
        depth *= torch.randn(N, 1).to(device) * self.noise_settings['depth'] + 1.0  # (N, 1)
        pt = utils3d.torch.unproject_cv(uv, depth, extrinsics_src, intrinsics).squeeze(-2)  # (N, 3)
        yaw = torch.randn(N).to(device) * self.noise_settings['yaw']  # (N)
        pitch = torch.randn(N).to(device) * self.noise_settings['pitch']  # (N)
        roll = torch.randn(N).to(device) * self.noise_settings['roll']  # (N)
        R = utils3d.torch.euler_angles_to_matrix(torch.stack([pitch, yaw, roll], dim=-1), 'ZXY')  # (N, 3, 3)
        T = pt + (R @ -pt.unsqueeze(-1)).squeeze(-1) * (torch.randn(N).to(device) * self.noise_settings['radius'] + 1.0).unsqueeze(-1)  # (N, 3)
        extrinsics_tgt = torch.eye(4, dtype=torch.float32).to(device).unsqueeze(0).repeat(N, 1, 1)  # (N, 4, 4)
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
        N, _, H, W = data['image'].shape
        device = data['image'].device
        uv = data[f'{self.uv_prefix}uv']
        mask = data[f'{self.uv_prefix}mask']
        uvm = torch.cat([uv, mask.float().unsqueeze(1)], dim=1)
        uvm = F.grid_sample(uvm, data[f'{self.out_prefix}uv'].permute(0, 2, 3, 1) * 2 - 1, mode='bilinear', padding_mode='zeros', align_corners=False)
        data[f'{self.out_prefix}flow'] = uvm
        data[f'{self.out_prefix}flow'][:, :2] -= utils3d.torch.image_uv(H, W).float().to(device).permute(2, 0, 1).unsqueeze(0)
        return data
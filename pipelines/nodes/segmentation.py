import torch
from typing import *

from utils.monodepth_utils import *
from utils.inpaint_utils import *
from utils.segment_utils import *
from .base import Node


class BackgroundRemoval(Node):
    def __init__(self, in_prefix: str = "", out_prefix: str = "rmbg_+"):
        super().__init__(in_prefix, out_prefix)
        self.model = load_u2net_model()

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Remove background.

        Args:
            image: (N, 3, H, W) tensor of images.
        Returns:
            rmbg_image: (N, 3, H, W) tensor of images.
            rmbg_mask: (N, H, W) tensor of masks.
        """
        rmbg_mask = u2net_predict_mask(self.model, data[f'{self.in_prefix}image'])
        data[f'{self.out_prefix}image'] = data[f'{self.in_prefix}image'] * rmbg_mask.unsqueeze(1)
        data[f'{self.out_prefix}mask'] = rmbg_mask
        return data


class ForegroundPoint(Node):
    def __init__(self, in_prefix: str = "rmbg_", out_prefix: str = ""):
        super().__init__(in_prefix, out_prefix)
    
    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Randomly sample foreground point.

        Args:
            rmbg_mask: (N, H, W) tensor of masks.
        Returns:
            point: (N, 2) tensor of points.
        """
        N, H, W = data[f'{self.in_prefix}mask'].shape
        device = data[f'{self.in_prefix}mask'].device
        mask = data[f'{self.in_prefix}mask'].reshape(N, -1)
        mask = mask / mask.sum(dim=-1, keepdim=True)
        point = torch.multinomial(mask, 1).float()
        point = torch.cat([point % W, point // W], dim=-1) / torch.tensor([W, H], dtype=torch.float32).to(device)
        data[f'{self.out_prefix}point'] = point
        return data


import torch
import torch.nn.functional as F
from typing import *

from .base import Node


class Resize(Node):
    def __init__(self, in_prefix: str = "", out_prefix: str = None, size: int = 128):
        super().__init__(in_prefix, out_prefix)
        self.size = size
    
    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Resize image.

        Args:
            image: (N, 3, H, W) tensor of images.
            depth: (N, H, W) tensor of depth.
        Returns:
            image: (N, 3, size, size) tensor of resized images.
            depth: (N, size, size) tensor of resized depth.
        """
        data[f'{self.out_prefix}image'] = F.interpolate(data[f'{self.in_prefix}image'], size=(self.size, self.size), mode='bicubic', align_corners=False, antialias=True)
        data[f'{self.out_prefix}depth'] = F.interpolate(data[f'{self.in_prefix}depth'].unsqueeze(1), size=(self.size, self.size), mode='nearest-exact').squeeze(1)
        return data


class Replicate(Node):
    def __init__(self, times: int = 2):
        super().__init__()
        self.times = times
    
    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Replicate image.

        Args:
            any: (N, C, H, W) or (N, H, W) tensor of anything.
        Returns:
            any: (N, C, H, W) or (N, H, W) tensor of replicated anything.
        """
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.unsqueeze(1).repeat(1, self.times, *([1] * (value.dim() - 1))).view(-1, *value.shape[1:])
        return data

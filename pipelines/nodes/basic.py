import torch
import torch.nn.functional as F
from typing import *

from utils.monodepth_utils import *
from utils.inpaint_utils import *
from utils.segment_utils import *
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
        Returns:
            image: (N, 3, size, size) tensor of resized images.
        """
        data[f'{self.out_prefix}image'] = F.interpolate(data[f'{self.in_prefix}image'], size=(self.size, self.size), mode='bicubic', align_corners=False, antialias=True)
        data[f'{self.out_prefix}depth'] = F.interpolate(data[f'{self.in_prefix}depth'].unsqueeze(1), size=(self.size, self.size), mode='nearest-exact').squeeze(1)
        return data

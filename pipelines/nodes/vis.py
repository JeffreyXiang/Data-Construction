import torch
from typing import *

from utils.monodepth_utils import *
from utils.inpaint_utils import *
from utils.segment_utils import *
from .base import Node


class PointVisualization(Node):
    def __init__(self, in_prefix: str = "", out_prefix: str = "+point_"):
        super().__init__(in_prefix, out_prefix)

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Show point.

        Args:
            image: (N, 3, H, W) tensor of images.
            point: (N, 2) tensor of points.
        Returns:
            point_image: (N, 3, H, W) tensor of images.
        """
        N, _, H, W = data[f'{self.in_prefix}image'].shape
        device = data[f'{self.in_prefix}image'].device
        pix_point = (data[f'{self.in_prefix}point'] * torch.tensor([W, H], dtype=torch.float32).to(device)).long()
        data[f'{self.out_prefix}image'] = data['image'].clone()
        r = max(1, min(H, W) // 128)
        for i in range(N):
            data[f'{self.out_prefix}image'][i, :, pix_point[i, 1]-r:pix_point[i, 1]+r, pix_point[i, 0]-r:pix_point[i, 0]+r] = torch.tensor([1, 0, 0], dtype=torch.float32).to(device)[:, None, None]
        return data


class BboxVisualization(Node):
    def __init__(self, in_prefix: str = "", out_prefix: str = "+bbox_"):
        super().__init__(in_prefix, out_prefix)

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Show bbox.

        Args:
            image: (N, 3, H, W) tensor of images.
            bbox: (N, 4) tensor of bbox.
        Returns:
            bbox_image: (N, 3, H, W) tensor of images.
        """
        N, _, H, W = data[f'{self.in_prefix}image'].shape
        device = data[f'{self.in_prefix}image'].device
        pix_bbox = data[f'{self.in_prefix}bbox'] * torch.tensor([W, H, W, H], dtype=torch.float32).to(device)
        pix_bbox = pix_bbox.long()
        data[f'{self.out_prefix}image'] = data['image'].clone()
        for i in range(N):
            data[f'{self.out_prefix}image'][i, :, pix_bbox[i, 1]:pix_bbox[i, 3], pix_bbox[i, 0]-1:pix_bbox[i, 0]+1] = torch.tensor([1, 0, 0], dtype=torch.float32).to(device)[:, None, None]
            data[f'{self.out_prefix}image'][i, :, pix_bbox[i, 1]:pix_bbox[i, 3], pix_bbox[i, 2]-1:pix_bbox[i, 2]+1] = torch.tensor([1, 0, 0], dtype=torch.float32).to(device)[:, None, None]
            data[f'{self.out_prefix}image'][i, :, pix_bbox[i, 1]-1:pix_bbox[i, 1]+1, pix_bbox[i, 0]:pix_bbox[i, 2]] = torch.tensor([1, 0, 0], dtype=torch.float32).to(device)[:, None, None]
            data[f'{self.out_prefix}image'][i, :, pix_bbox[i, 3]-1:pix_bbox[i, 3]+1, pix_bbox[i, 0]:pix_bbox[i, 2]] = torch.tensor([1, 0, 0], dtype=torch.float32).to(device)[:, None, None]
        return data


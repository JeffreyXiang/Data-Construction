import torch
from typing import *

from utils.monodepth_utils import *
from .base import Node


class DepthPrediction(Node):
    MODEL_REGISTRY = {
        'zoedepth': (load_zoedepth_model, zoedepth_predict_depth),
    }

    def __init__(self, in_prefix: str = "", out_prefix: str = None, model_name: str = 'zoedepth'):
        super().__init__(in_prefix, out_prefix)
        self.model_name = model_name
        self.model = self.MODEL_REGISTRY[self.model_name][0]()
        self.predict_depth = self.MODEL_REGISTRY[self.model_name][1]
    
    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Predict depth from image using ZoeDepth.

        Args:
            image: (N, 3, H, W) tensor of images.
        Returns:
            depth: (N, H, W) tensor of metric depth.
        """
        depth_tensor = self.predict_depth(self.model, data[f'{self.in_prefix}image'])
        data[f'{self.out_prefix}depth'] = depth_tensor
        return data


class DisparityPrediction(Node):
    MODEL_REGISTRY = {
        # 'midas': (load_midas_model, midas_predict_disparity),
        'depth_anything': (load_depthanything_model, depthanything_predict_disparity),
    }

    def __init__(self, in_prefix: str = "", out_prefix: str = None, model_name: str = 'depth_anything'):
        super().__init__(in_prefix, out_prefix)
        self.model_name = model_name
        self.model = self.MODEL_REGISTRY[self.model_name][0]()
        self.predict_disparity = self.MODEL_REGISTRY[self.model_name][1]

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Predict depth from image using DepthAnything.

        Args:
            image: (N, 3, H, W) tensor of images.
        Returns:
            disparity: (N, H, W) tensor of disparity.
        """
        disparity_tensor = self.predict_disparity(self.model, data[f'{self.in_prefix}image'])
        data[f'{self.out_prefix}disparity'] = disparity_tensor
        return data


class DepthAlignment(Node):
    def __init__(self, in_prefix: List[str] = ["inpainted_warped_", "warped_"], out_prefix: str = None):
        super().__init__(in_prefix, out_prefix)

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Align depth.

        Args:
            warped_depth: (N, H, W) tensor of warped depth.
            warped_mask: (N, H, W) tensor of warped mask.
            inpainted_warped_depth: (N, H, W) tensor of inpainted warped depth.
        Returns:
            inpainted_warped_depth: (N, H, W) tensor of aligned depth.
        """
        mean_first = data[f'{self.in_prefix[1]}depth'][data[f'{self.in_prefix[1]}mask']].mean()
        mean_second = data[f'{self.in_prefix[0]}depth'][data[f'{self.in_prefix[1]}mask']].mean()
        data[f'{self.out_prefix}depth'] *= mean_first / mean_second
        return data
    


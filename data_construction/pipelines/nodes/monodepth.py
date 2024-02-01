import torch
from typing import *

from ...utils.monodepth_utils import *
from .base import Node


class DepthPrediction(Node):
    MODEL_REGISTRY = {
        'zoedepth': (load_zoedepth_model, zoedepth_predict_depth),
    }

    def __init__(self, in_prefix: str = "", out_prefix: str = None, model_name: str = 'zoedepth', batch_size: int = 8):
        super().__init__(in_prefix, out_prefix)
        self.model_name = model_name
        self._model_fn, self._predict_fn = self.MODEL_REGISTRY[self.model_name]
        self.model = None
        self.batch_size = batch_size
    
    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Predict depth from image using ZoeDepth.

        Args:
            image: (N, 3, H, W) tensor of images.
        Returns:
            depth: (N, H, W) tensor of metric depth.
        """
        N = data[f'{self.in_prefix}image'].shape[0]
        if self.model is None:
            self.model = pipe.get_shared_component(self.model_name, self._model_fn)
        depth_tensor = []
        for i in range(0, N, self.batch_size):
            depth_tensor.append(self._predict_fn(self.model, data[f'{self.in_prefix}image'][i:i+self.batch_size]))
        depth_tensor = torch.cat(depth_tensor, dim=0)
        data[f'{self.out_prefix}depth'] = depth_tensor
        return data


class DisparityPrediction(Node):
    MODEL_REGISTRY = {
        # 'midas': (load_midas_model, midas_predict_disparity),
        'depth_anything': (load_depthanything_model, depthanything_predict_disparity),
    }

    def __init__(self, in_prefix: str = "", out_prefix: str = None, model_name: str = 'depth_anything', batch_size: int = 8):
        super().__init__(in_prefix, out_prefix)
        self.model_name = model_name
        self._model_fn, self._predict_fn = self.MODEL_REGISTRY[self.model_name]
        self.model = None
        self.batch_size = batch_size

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Predict depth from image using DepthAnything.

        Args:
            image: (N, 3, H, W) tensor of images.
        Returns:
            disparity: (N, H, W) tensor of disparity.
        """
        N = data[f'{self.in_prefix}image'].shape[0]
        if self.model is None:
            self.model = pipe.get_shared_component(self.model_name, self._model_fn)
        disparity_tensor = []
        for i in range(0, N, self.batch_size):
            disparity_tensor.append(self._predict_fn(self.model, data[f'{self.in_prefix}image'][i:i+self.batch_size]))
        disparity_tensor = torch.cat(disparity_tensor, dim=0)
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
    


import torch
from typing import *

from ...utils.inpaint_utils import *
from .base import Node

    
class Inpainting(Node):
    def __init__(self, in_prefix: str = "warped_", out_prefix: str = "inpainted_+", batch_size: int = 8):
        super().__init__(in_prefix, out_prefix)
        self.batch_size = batch_size
    
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Inpainting.

        Args:
            image: (N, 3, H, W) tensor of warped images.
            depth: (N, H, W) tensor of warped depth.
            mask: (N, H, W) tensor of warped mask.
        Returns:
            inpainted_image: (N, 3, H, W) tensor of inpainted warped images.
        """
        N = data[f'{self.in_prefix}image'].shape[0]
        model = self.get_lazy_component('sdv2_inpaint', load_sdv2_inpaint_model, pipe=pipe)
        inpainted_image = []
        for i in range(0, N, self.batch_size):
            inpainted_image.append(sdv2_inpaint(model, data[f'{self.in_prefix}image'][i:i+self.batch_size], ~data[f'{self.in_prefix}mask'][i:i+self.batch_size]))
        inpainted_image = torch.cat(inpainted_image, dim=0)
        data[f'{self.out_prefix}image'] = inpainted_image
        return data


class IvidInpainting(Node):
    def __init__(self, in_prefix: str = "warped_", out_prefix: str = "inpainted_+", batch_size: int = 8):
        super().__init__(in_prefix, out_prefix)
        self.batch_size = batch_size
    
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Inpainting.

        Args:
            image: (N, 3, H, W) tensor of warped images.
            depth: (N, H, W) tensor of warped depth.
            mask: (N, H, W) tensor of warped mask.
        Returns:
            inpainted_image: (N, 3, H, W) tensor of inpainted warped images.
        """
        N = data[f'{self.in_prefix}image'].shape[0]
        model = self.get_lazy_component('ivid_inpaint', load_ivid_inpaint_model, pipe=pipe)
        inpainted_image = []
        for i in range(0, N, self.batch_size):
            inpainted_image.append(ivid_inpaint(model, data[f'{self.in_prefix}image'][i:i+self.batch_size], data[f'{self.in_prefix}depth'][i:i+self.batch_size], data[f'{self.in_prefix}mask'][i:i+self.batch_size]))
        inpainted_image = torch.cat(inpainted_image, dim=0)
        data[f'{self.out_prefix}image'] = inpainted_image
        return data

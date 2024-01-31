import torch
from typing import *

from utils.inpaint_utils import *
from .base import Node

    
class Inpainting(Node):
    def __init__(self, in_prefix: str = "warped_", out_prefix: str = "inpainted_+"):
        super().__init__(in_prefix, out_prefix)
        self.model = load_sdv2_inpaint_model()
    
    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Inpainting.

        Args:
            image: (N, 3, H, W) tensor of warped images.
            depth: (N, H, W) tensor of warped depth.
            mask: (N, H, W) tensor of warped mask.
        Returns:
            inpainted_image: (N, 3, H, W) tensor of inpainted warped images.
        """
        inpainted_image = sdv2_inpaint(self.model, data[f'{self.in_prefix}image'], ~data[f'{self.in_prefix}mask'])
        data[f'{self.out_prefix}image'] = inpainted_image
        return data


class IvidInpainting(Node):
    def __init__(self, in_prefix: str = "warped_", out_prefix: str = "inpainted_+"):
        super().__init__(in_prefix, out_prefix)
        self.model = load_ivid_inpaint_model()
    
    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Inpainting.

        Args:
            image: (N, 3, H, W) tensor of warped images.
            depth: (N, H, W) tensor of warped depth.
            mask: (N, H, W) tensor of warped mask.
        Returns:
            inpainted_image: (N, 3, H, W) tensor of inpainted warped images.
        """
        inpainted_image = ivid_inpaint(self.model, data[f'{self.in_prefix}image'], data[f'{self.in_prefix}depth'], data[f'{self.in_prefix}mask'])
        data[f'{self.out_prefix}image'] = inpainted_image
        return data

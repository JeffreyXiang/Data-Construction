import torch
from ..models import rmbg

__all__ = [
    'load_u2net_model',
    'u2net_predict_mask',
]


def load_u2net_model(model_name='u2net', device='cuda'):
    return rmbg.load(model_name, device)


@torch.no_grad()
def u2net_predict_mask(net, image: torch.Tensor):
    """
    Predict mask from image using U2NET.

    Args:
        image: (N, 3, H, W) tensor of images.
    Returns:
        mask: (N, H, W) tensor of masks.
    """
    return rmbg.predict_mask(net, image)

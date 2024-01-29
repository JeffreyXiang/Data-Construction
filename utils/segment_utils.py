import torch
from models import rmbg

def load_u2net_model(model_name='u2net'):
    return rmbg.load(model_name)


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

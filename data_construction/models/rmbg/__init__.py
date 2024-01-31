import os
import torch
from torch.autograd import Variable

from .u2net import U2NET # full size version 173.6 MB
from .u2net import U2NETP # small version u2net 4.7 MB
from .utils import *


def load(model_name='u2net', device='cuda'):
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    path = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.pt')
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()
    net.to(device)
    return net


@torch.no_grad()
def predict_mask(net, image: torch.Tensor):
    """
    Predict mask from image using U2NET.

    Args:
        image: (N, 3, H, W) tensor of images.
    Returns:
        mask: (N, H, W) tensor of masks.
    """
    image = preprocess_image(image)
    image = Variable(image.float().cuda())
    d1,d2,d3,d4,d5,d6,d7= net(image)
    pred = d1[:,0,:,:]
    pred = norm_pred(pred)
    return pred


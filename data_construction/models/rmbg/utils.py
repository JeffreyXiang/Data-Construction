import torch
import torch.nn.functional as F


def preprocess_image(image):
    N, C, H, W = image.shape
    image = image / torch.max(image.reshape(N, -1), dim=1)[0].reshape(N, 1, 1, 1)
    image = (image - 0.485) / 0.229
    return image


# normalize the predicted SOD probability map
def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

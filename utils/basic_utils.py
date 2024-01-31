import os
import hashlib
import numpy as np
import io
import cv2
from PIL import Image
import imageio
import OpenEXR as exr
import Imath as exr_types
import pickle
from torchvision import utils


def pack_image(image: np.ndarray) -> bytes:
    """
    Pack image to bytes.

    Args:
        image (np.ndarray): Image to pack.
    Returns:
        data (bytes): Packed image.
    """
    H, W, C = image.shape if len(image.shape) == 3 else (*image.shape, 1)
    image = image.reshape(H, W, C)
    dtype = image.dtype
    if dtype == np.uint8:
        fp = io.BytesIO()
        imageio.imwrite(fp, image, format='png')
        return fp.getvalue()
    elif dtype == np.float16:
        header = exr.Header(W, H)
        header['channels'] = {ch: exr_types.Channel(exr_types.PixelType(exr.HALF)) for ch in 'RGBA'[:C]}
        header['compression'] = exr_types.Compression(exr_types.Compression.ZIP_COMPRESSION)
        # hash with timestamp to avoid collision
        filename = os.path.join(os.path.dirname(__file__), 'tmp', f'{hashlib.md5(image.tobytes()).hexdigest()}.exr')
        out = exr.OutputFile(filename, header)
        out.writePixels({ch: image[:, :, i].astype(np.float16).tobytes() for i, ch in enumerate('RGBA'[:C])})
        out.close()
        with open(filename, 'rb') as f:
            data = f.read()
        os.remove(filename)
        return data
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")
    

def resize_image(
        image: np.ndarray,
        size: tuple | int,
        resize_mode: str = 'shorter',
        resample_mode: int = Image.LANCZOS
    ) -> np.ndarray:
    """
    Resize image to a given size.

    Args:
        image (np.ndarray): Image to resize.
        size (tuple | int): Size to resize to. If int, resize according to the
            resize_mode.
        resize_mode (str): How to determine the size to resize to. Can be
            'shorter', 'longer', or 'square'.
        resample_mode (int): Resampling mode. See PIL.Image.resize for details.
    Returns:
        image (np.ndarray): Resized image.
    """
    assert len(image.shape) == 3, "Image must be 3-dimensional."
    assert image.dtype == np.uint8, "Image must be uint8."

    H, W, C = image.shape
    if isinstance(size, int):
        if resize_mode == 'shorter':
            scale = size / min(H, W)
            size = (int(W * scale), int(H * scale))
        elif resize_mode == 'longer':
            scale = size / max(H, W)
            size = (int(W * scale), int(H * scale))
        elif resize_mode == 'square':
            size = (size, size)
        else:
            raise NotImplementedError(f"Unsupported size mode {resize_mode}")

    image = Image.fromarray(image)
    image = image.resize(size, resample_mode)
    image = np.array(image)

    return image


def pad_image(
        image: np.ndarray,
        size: tuple | int,
        resize_mode: str = 'none',
        resample_mode: int = Image.LANCZOS,
        pad_mode: str = 'constant',
        pad_value: int | tuple = 0
    ) -> np.ndarray:
    """
    Pad image to a given size.

    Args:
        image (np.ndarray): Image to pad.
        size (tuple | int): Size to pad to. If int, pad to a square.
        resize_mode (str): How to determine the size to resize to. Can be
            'none', 'max_fit'
        resample_mode (int): Resampling mode. See PIL.Image.resize for details.
        pad_mode (str): Padding mode. See np.pad for details.
        pad_value (int | tuple): Padding value. See np.pad for details.
    Returns:
        image (np.ndarray): Padded image.
    """
    assert len(image.shape) == 3, "Image must be 3-dimensional."
    assert image.dtype == np.uint8, "Image must be uint8."

    H, W, C = image.shape
    if isinstance(size, int):
        size = (size, size)
    if resize_mode == 'none':
        if H > size[0] or W > size[1]:
            raise ValueError(f"Image size ({H}, {W}) is larger than target size ({size[0]}, {size[1]}).")
    elif resize_mode == 'max_fit':
        scale = min(size[0] / H, size[1] / W)
        image = resize_image(image, (int(W * scale), int(H * scale)), resample_mode=resample_mode)
        H, W, C = image.shape
    pad_top = (size[0] - H) // 2
    pad_bottom = size[0] - H - pad_top
    pad_left = (size[1] - W) // 2
    pad_right = size[1] - W - pad_left
    image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), pad_mode, constant_values=pad_value)

    return image


def dilate_mask(mask, struct_elem):
    return cv2.dilate(mask.astype(np.uint8), struct_elem) > 0


def erode_mask(mask, struct_elem):
    return cv2.erode(mask.astype(np.uint8), struct_elem) > 0


def _create_sdf_field(mask, radius=10):
    sdf_field = mask.copy().astype(np.float32)
    sdf_field[mask] = -radius - 0.5
    sdf_field[~mask] = radius + 0.5
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(radius):
        if i == 0:
            cur_mask = (dilate_mask(mask, struct_elem) != erode_mask(mask, struct_elem))
        else:
            cur_mask = dilate_mask(cur_mask, struct_elem)
        sdf_field[cur_mask & mask] = -i - 0.5
        sdf_field[cur_mask & ~mask] = i + 0.5
    return sdf_field


def soften_mask(mask, radius=10):
    sdf_field = _create_sdf_field(mask, radius=radius)
    k_size = 2 * radius + 1
    sdf_field = cv2.GaussianBlur(sdf_field, (k_size, k_size), sigmaX=0, borderType=cv2.BORDER_DEFAULT)
    new_mask = np.clip(-sdf_field + 0.5, 0, 1)
    return new_mask


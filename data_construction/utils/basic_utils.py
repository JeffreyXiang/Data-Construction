import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from typing import *

import numpy as np
import cv2
from PIL import Image
import zipfile
from typing import *



__all__ = [
    # Dictionary utils
    'dict_merge',
    'dict_foreach',
    'dict_reduce',
    'dict_any',
    'dict_all',
    'dict_flatten',
    # Image utils
    'pack_image',
    'unpack_image',
    'resize_image',
    'pad_image',
    'dilate_mask',
    'erode_mask',
    'soften_mask',
    # Save utils
    'dict_save_zip',
]


# Dictionary utils
def _dict_merge(dicta, dictb, prefix=''):
    """
    Merge two dictionaries.
    """
    assert isinstance(dicta, dict), 'input must be a dictionary'
    assert isinstance(dictb, dict), 'input must be a dictionary'
    dict_ = {}
    all_keys = set(dicta.keys()).union(set(dictb.keys()))
    for key in all_keys:
        if key in dicta.keys() and key in dictb.keys():
            if isinstance(dicta[key], dict) and isinstance(dictb[key], dict):
                dict_[key] = _dict_merge(dicta[key], dictb[key], prefix=f'{prefix}.{key}')
            else:
                raise ValueError(f'Duplicate key {prefix}.{key} found in both dictionaries. Types: {type(dicta[key])}, {type(dictb[key])}')
        elif key in dicta.keys():
            dict_[key] = dicta[key]
        else:
            dict_[key] = dictb[key]
    return dict_


def dict_merge(dicta, dictb):
    """
    Merge two dictionaries.
    """
    return _dict_merge(dicta, dictb, prefix='')


def dict_foreach(dic, func, special_func={}):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            dic[key] = dict_foreach(dic[key], func)
        else:
            if key in special_func.keys():
                dic[key] = special_func[key](dic[key])
            else:
                dic[key] = func(dic[key])
    return dic


def dict_reduce(dicts, func, special_func={}):
    """
    Reduce a list of dictionaries. Leaf values must be scalars.
    """
    assert isinstance(dicts, list), 'input must be a list of dictionaries'
    assert all([isinstance(d, dict) for d in dicts]), 'input must be a list of dictionaries'
    assert len(dicts) > 0, 'input must be a non-empty list of dictionaries'
    all_keys = set([key for dict_ in dicts for key in dict_.keys()])
    reduced_dict = {}
    for key in all_keys:
        vlist = [dict_[key] for dict_ in dicts if key in dict_.keys()]
        if isinstance(vlist[0], dict):
            reduced_dict[key] = dict_reduce(vlist, func, special_func)
        else:
            if key in special_func.keys():
                reduced_dict[key] = special_func[key](vlist)
            else:
                reduced_dict[key] = func(vlist)
    return reduced_dict


def dict_any(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if dict_any(dic[key], func):
                return True
        else:
            if func(dic[key]):
                return True
    return False


def dict_all(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if not dict_all(dic[key], func):
                return False
        else:
            if not func(dic[key]):
                return False
    return True


def dict_flatten(dic, sep='.'):
    """
    Flatten a nested dictionary into a dictionary with no nested dictionaries.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    flat_dict = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            sub_dict = dict_flatten(dic[key], sep=sep)
            for sub_key in sub_dict.keys():
                flat_dict[key + sep + sub_key] = sub_dict[sub_key]
        else:
            flat_dict[key] = dic[key]
    return flat_dict


# Image utils
def pack_image(image: np.ndarray) -> bytes:
    """
    Pack image to bytes. NOTE: Input channels are in RGBA order.

    Args:
        image (np.ndarray): Image to pack.
    Returns:
        data (bytes): Packed image.
    """
    C = image.shape[2] if image.ndim == 3 else 1
    assert C in [1, 3, 4], "Image must have 1, 3, or 4 channels."
    # RGBA -> BGRA
    if C == 4:
        image = image[..., [2, 1, 0, 3]]
    elif C == 3:
        image = image[..., [2, 1, 0]]

    dtype = image.dtype
    if dtype == np.bool_:
        flag, data = cv2.imencode('.png', image.astype(np.uint8) * 255, (cv2.IMWRITE_PNG_BILEVEL, 1))
    elif dtype == np.uint8 or dtype == np.uint16:
        flag, data = cv2.imencode('.png', image, (cv2.IMWRITE_PNG_COMPRESSION, 9))
    elif dtype == np.float16:
        flag, data = cv2.imencode('.exr', image.astype(np.float32), (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF))
    elif dtype == np.float32:
        flag, data = cv2.imencode('.exr', image, (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    if not flag:
        raise ValueError("Failed to encode image.")
    return data.tobytes()
    
    
def unpack_image(data: bytes) -> np.ndarray:
    """
    Unpack image from bytes. NOTE: Output channels are in RGBA order.

    Args:
        data (bytes): Packed image.
    Returns:
        image (np.ndarray): Unpacked image.
    """
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode image.")
    
    C = image.shape[2] if image.ndim == 3 else 1
    # BGRA -> RGBA
    if C == 4:
        image = image[..., [2, 1, 0, 3]]
    elif C == 3:
        image = image[..., [2, 1, 0]]

    return image
    

def resize_image(
        image: np.ndarray,
        size: Union[tuple, int],
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
        size: Union[tuple, int],
        resize_mode: str = 'none',
        resample_mode: int = Image.LANCZOS,
        pad_mode: str = 'constant',
        pad_value: Union[int, tuple] = 0
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


# Save utils
def dict_save_zip(data, filename):
    """
    Save a dictionary to a zip file.
    The leaf values of the dictionary can only be bytes or strings.
    """
    data = dict_flatten(data, sep='/')
    with zipfile.ZipFile(filename, 'w') as f:
        for k, v in data.items():
            assert isinstance(v, (bytes, str)), f"Leaf value must be bytes or string. Got {type(v)} for key {k}"
            f.writestr(k, v)

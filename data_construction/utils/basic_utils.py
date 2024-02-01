import numpy as np
import io
import cv2
from PIL import Image
import imageio
import OpenEXR as exr
import Imath as exr_types
import tempfile
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
        if C == 1:
            image = image.squeeze(-1)
        imageio.imwrite(fp, image, format='png')
        return fp.getvalue()
    elif dtype == np.float16:
        header = exr.Header(W, H)
        header['channels'] = {ch: exr_types.Channel(exr_types.PixelType(exr.HALF)) for ch in 'RGBA'[:C]}
        header['compression'] = exr_types.Compression(exr_types.Compression.ZIP_COMPRESSION)
        with tempfile.NamedTemporaryFile() as f:
            filename = f.name
            out = exr.OutputFile(filename, header)
            out.writePixels({ch: image[:, :, i].astype(np.float16).tobytes() for i, ch in enumerate('RGBA'[:C])})
            out.close()
            with open(filename, 'rb') as f:
                data = f.read()
        return data
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    
def unpack_image(data: bytes) -> np.ndarray:
    """
    Unpack image from bytes.

    Args:
        data (bytes): Packed image.
    Returns:
        image (np.ndarray): Unpacked image.
    """
    SUPPORTED_HEADER = {
        'png': b'\x89PNG\r\n\x1a\n',
        'exr': b'\x76\x2f\x31\x01',
    }

    type_ = 'unknown'
    for ext, header in SUPPORTED_HEADER.items():
        if data[:len(header)] == header:
            type_ = ext
            break

    if type_ == 'png':
        image = imageio.imread(io.BytesIO(data))
    elif type_ == 'exr':
        with tempfile.NamedTemporaryFile() as f:
            f.write(data)
            f.seek(0)
            dr = exr.InputFile(f.name)
            header = dr.header()
            W, H = header['dataWindow'].max.x - header['dataWindow'].min.x + 1, header['dataWindow'].max.y - header['dataWindow'].min.y + 1
            C = len(header['channels'])
            image = np.zeros((H, W, C), dtype=np.float16)
            for i, ch in enumerate('RGBA'[:C]):
                image[:, :, i] = np.frombuffer(dr.channel(ch), dtype=np.float16).reshape(H, W)
            if C == 1:
                image = image.squeeze(-1)
    else:
        raise NotImplementedError(f"Unsupported image type")

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

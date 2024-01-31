from abc import abstractmethod
import os
import json
import random
import torch
from torchvision import transforms
import PIL.Image as Image
from typing import *


class SA_1B(torch.utils.data.Dataset):
    """
    Dataset for images.
    """
    def __init__(
            self,
            root: str,
            image_size: int = 512,
            crop: bool = False,
            normalize: bool = False,
            return_annotation: bool = False,
        ):
        """
        Args:
            root: Root directory of dataset.
            transform: Transform to apply to image.
        """
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.crop = crop
        self.normalize = normalize
        self.return_annotation = return_annotation
        self.filenames = os.listdir(os.path.join(root, "images"))
        self.filenames = [os.path.splitext(filename)[0] for filename in self.filenames]
        self.filenames = sorted(self.filenames)
        
    def __len__(self):
        return len(self.filenames)
    
    def _is_background_bbox(self, bbox, size):
        """
        Check if a bbox is background.
        """
        x1, y1, x2, y2 = bbox
        w, h = size
        hb = y2 - y1
        wb = x2 - x1
        if hb > 0.8 * h and wb > 0.8 * w:
            return True
        else:
            return False

    def _crop_image(self, image, annotations):
        # Filter out background
        annotations = [anno for anno in annotations if not self._is_background_bbox(anno['crop_box'], image.size)]

        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        threshold = 0.5 * annotations[0]['area']
        annotations = [anno for anno in annotations if anno['area'] >= threshold]
        anno = random.choice(annotations)
        mask = anno['bbox']

        # Crop image
        ## Crop box size
        x1 = int(mask[0])
        y1 = int(mask[1])
        x2 = int(mask[0] + mask[2])
        y2 = int(mask[1] + mask[3])
        assert x1 < x2 and y1 < y2 and 0 <= x1 < image.size[0] and 0 <= x2 < image.size[0] and 0 <= y1 < image.size[1] and 0 <= y2 < image.size[1], f"Invalid bbox: {mask} in {image.size}"
        size_bbox = max(y2 - y1, x2 - x1)
        size_min = min(max(size_bbox, self.image_size), min(image.size))
        size_max = max(min(4 * size_bbox, min(image.size)), self.image_size)
        size = random.randint(size_min, size_max)

        ## Crop box offset (upper left corner)
        left_min = min(max(0, x2 - size), x1)
        left_max = max(min(x1, image.size[0] - size), x2 - size)
        assert left_min <= left_max, f"Invalid crop box with size {size} and bbox {mask} in {image.size}"
        left = random.randint(left_min, left_max)
        top_min = min(max(0, y2 - size), y1)
        top_max = max(min(y1, image.size[1] - size), y2 - size)
        assert top_min <= top_max, f"Invalid crop box with size {size} and bbox {mask} in {image.size}"
        top = random.randint(top_min, top_max)

        ## Crop
        image = image.crop((left, top, left + size, top + size))
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        point = torch.tensor([
            (anno['point_coords'][0][0] - left) / size,
            (anno['point_coords'][0][1] - top) / size,
        ], dtype=torch.float32)
        bbox = torch.tensor([
            max(0, x1 - left) / size,
            max(0, y1 - top) / size,
            min(size, x2 - left) / size,
            min(size, y2 - top) / size,
        ], dtype=torch.float32)

        return image, point, bbox
    
    def __getitem__(self, index: int):
        filename = self.filenames[index]
        image_path = os.path.join(self.root, "images", filename + ".jpg")

        ret = {'filename': filename}
            
        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.return_annotation or self.crop:
            annotation_path = os.path.join(self.root, "annotations", filename + ".json")
            with open(annotation_path) as f:
                annotations = json.load(f)['annotations']
            for i in range(len(annotations)):
                annotations[i]['bbox'] = [int(x) for x in annotations[i]['bbox']]
                annotations[i]['crop_box'] = [int(x) for x in annotations[i]['crop_box']]
        
        if self.crop:
            image, point, bbox = self._crop_image(image, annotations)
            ret['point'] = point
            ret['bbox'] = bbox

        if self.return_annotation:
            ret['annotations'] = json.dumps(annotations)

        # Normalize
        image = transforms.ToTensor()(image)
        if self.normalize:
            image = image * 2 - 1

        ret['image'] = image

        return ret

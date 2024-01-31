import os
import numpy as np
import torch
from torchvision import transforms
import PIL.Image as Image
from typing import *
import utils3d


class ObjectImageDataset(torch.utils.data.Dataset):
    """
    Dataset for images.
    """
    def __init__(self, root: str, image_size: int = 128, normalize: bool = True):
        """
        Args:
            root: Root directory of dataset.
            transform: Transform to apply to image.
        """
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.filenames = os.listdir(os.path.join(root, "images"))
        self.filenames = [os.path.splitext(filename)[0] for filename in self.filenames]
        self.filenames = sorted(self.filenames)
        self.normalize = normalize
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index: int):
        filename = self.filenames[index]
        image_path = os.path.join(self.root, "images", filename + ".jpg")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Crop image
        w, h = image.size
        cx = w // 2
        cy = h // 2
        hsize = min(w, h) // 2
        image = image.crop((cx - hsize, cy - hsize, cx + hsize, cy + hsize))

        # Resize image
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Normalize
        image = transforms.ToTensor()(image)
        if self.normalize:
            image = image * 2 - 1

        return {
            'filename': filename,
            'image': image,
        }


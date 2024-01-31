import torch
import numpy as np
from typing import *
import cv2
import json
from skimage.filters import rank  
from skimage.morphology import disk  
from pycocotools import mask as mask_utils

from ...utils.segment_utils import *
from ...models import segment_anything
from .base import Node


class BackgroundRemoval(Node):
    def __init__(self, in_prefix: str = "", out_prefix: str = "rmbg_+"):
        super().__init__(in_prefix, out_prefix)
        self.model = load_u2net_model()

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Remove background.

        Args:
            image: (N, 3, H, W) tensor of images.
        Returns:
            rmbg_image: (N, 3, H, W) tensor of images.
            rmbg_mask: (N, H, W) tensor of masks.
        """
        rmbg_mask = u2net_predict_mask(self.model, data[f'{self.in_prefix}image'])
        data[f'{self.out_prefix}image'] = data[f'{self.in_prefix}image'] * rmbg_mask.unsqueeze(1)
        data[f'{self.out_prefix}mask'] = rmbg_mask
        return data


class ForegroundPoint(Node):
    def __init__(self, in_prefix: str = "rmbg_", out_prefix: str = ""):
        super().__init__(in_prefix, out_prefix)
    
    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Randomly sample foreground point.

        Args:
            rmbg_mask: (N, H, W) tensor of masks.
        Returns:
            point: (N, 2) tensor of points.
        """
        N, H, W = data[f'{self.in_prefix}mask'].shape
        device = data[f'{self.in_prefix}mask'].device
        mask = data[f'{self.in_prefix}mask'].reshape(N, -1)
        mask = mask / mask.sum(dim=-1, keepdim=True)
        point = torch.multinomial(mask, 1).float()
        point = torch.cat([point % W, point // W], dim=-1) / torch.tensor([W, H], dtype=torch.float32).to(device)
        data[f'{self.out_prefix}point'] = point
        return data
    

class SegmentAnything(Node):
    def __init__(self, in_prefix: str = "", out_prefix: str = ""):
        super().__init__(in_prefix, out_prefix)
        self.model = segment_anything.load()
        self.segmentor = segment_anything.SamAutomaticMaskGenerator(self.model)

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Segment anything.

        Args:
            image: (N, 3, H, W) tensor of images.
        Returns:
            seg_masks: list of MaskData
        """
        seg_masks = []
        for i in data[f'{self.in_prefix}image']:
            seg_masks.append(self.segmentor.generate(i))
        data[f'{self.out_prefix}masks'] = seg_masks
        return data


class AnnotationToMask(Node):
    def __init__(self, in_prefix: str = "", out_prefix: str = ""):
        super().__init__(in_prefix, out_prefix)

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Convert annotation to mask.

        Args:
            annotations: list of dict
        Returns:
            masks: list of MaskData
        """
        masks = []
        for annos in data[f'{self.in_prefix}annotations']:
            annos = json.loads(annos)
            for i in range(len(annos)):
                annos[i]['segmentation'] = mask_utils.decode(annos[i]['segmentation'])
            masks.append(annos)
            
        data[f'{self.out_prefix}masks'] = masks
        return data


class ObjectMaskFiltering(Node):
    def __init__(
            self,
            in_prefix: str = "",
            out_prefix: str = "objseg_",
            depth_dilate_radius: int = 16,
            depth_far_thresh: float = 0.5,
            depth_edge_thresh: float = 0.05,
            depth_occluder_ratio_thresh: float = 0.01,
            depth_background_ratio_thresh: float = 0.75,
            min_mask_region_size=512,
        ):
        super().__init__(in_prefix, out_prefix)
        self.depth_dilate_radius = depth_dilate_radius
        self.depth_far_thresh = depth_far_thresh
        self.depth_edge_thresh = depth_edge_thresh
        self.depth_occluder_ratio_thresh = depth_occluder_ratio_thresh
        self.depth_background_ratio_thresh = depth_background_ratio_thresh
        self.min_mask_region_size = min_mask_region_size

    def _postprocess_depth_occlusion(
        self, masks: list, depth: torch.Tensor,
    ) -> list:
        """
        Removes masks that are occluded by its neighbors in depth.
        """
        if len(masks) == 0:
            return masks
        
        # Filter masks by depth occlusion
        depth = depth.cpu().numpy()
        depth = depth / np.max(depth)
        keep = []
        for idx in range(len(masks)):
            mask = masks[idx]['segmentation']
            bbox = masks[idx]['bbox']
            H, W = mask.shape
            ## pre crop mask
            C_L = max(0, bbox[0] - self.depth_dilate_radius - 16)
            C_R = min(W, bbox[0] + bbox[2] + self.depth_dilate_radius + 16)
            C_U = max(0, bbox[1] - self.depth_dilate_radius - 16)
            C_D = min(H, bbox[1] + bbox[3] + self.depth_dilate_radius + 16)
            mask_crop = mask[C_U:C_D, C_L:C_R]
            depth_crop = depth[C_U:C_D, C_L:C_R]

            # filter out masks that are too small
            if max(bbox[2], bbox[3]) < self.min_mask_region_size:
                continue

            # filter out masks that are too close to the image edge
            if bbox[0] < W // 32 or bbox[1] < H // 32 or bbox[0] + bbox[2] > W * 31 // 32 or bbox[1] + bbox[3] > H * 31 // 32:
                continue

            # filter out masks that contain more than one connected component
            num, _ = cv2.connectedComponents(mask_crop.astype(np.uint8))
            if num > 2:
                continue

            # filter out masks that are too far away
            if np.sum(depth_crop * mask_crop) / np.sum(mask_crop) < self.depth_far_thresh:
                continue

            # filter out masks whose neighbors are closer to the camera
            k_size = 2 * self.depth_dilate_radius + 1
            ## median filter depth
            depth_masked = np.where(mask_crop, depth_crop, 0)
            depth_masked = (depth_masked * 255).astype(np.uint8)
            depth_filtered = rank.median(depth_masked, disk(self.depth_dilate_radius), mask=mask_crop) / 255
            depth_filtered = np.where(mask_crop, depth_filtered, depth_crop)
            mask_dilated = cv2.dilate(mask_crop.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)))
            mask_dilated_eroded = cv2.erode(mask_dilated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size + 4, k_size + 4)))
            contour = (mask_crop & ~mask_dilated_eroded).astype(np.bool_)
            depth_masked = np.where(mask_dilated, 0, depth_crop)
            depth_masked_dilated = cv2.dilate(depth_masked, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size + 4, k_size + 4)))
            occluder_sum = np.sum((depth_masked_dilated[contour] - depth_filtered[contour]) > self.depth_edge_thresh)
            background_sum = np.sum((depth_filtered[contour] - depth_masked_dilated[contour]) > self.depth_edge_thresh)
            total_num = np.sum(contour)
            if (occluder_sum / total_num) > self.depth_occluder_ratio_thresh or (background_sum / total_num) < self.depth_background_ratio_thresh:
                continue

            keep.append(idx)

        # Filter masks
        masks = [masks[i] for i in keep]

        return masks

    def _nms(self, masks: list) -> list:
        """
        Perform non-maximum suppression on masks.
        """
        if len(masks) == 0:
            return masks

        masks = sorted(masks, key=lambda x: np.sum(x['segmentation']), reverse=True)
        keep = np.ones(len(masks), dtype=np.bool_)
        for i in range(len(masks)):
            if keep[i] == False:
                continue
            for j in range(i + 1, len(masks)):
                ratio = np.sum(masks[i]['segmentation'] * masks[j]['segmentation']) / np.sum(masks[j]['segmentation'])
                if ratio > 0.7:
                    keep[j] = False
        keep = np.where(keep)[0]
        return [masks[i] for i in keep]

    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        """
        Segment object.

        Args:
            masks: list of MaskData
            disparity: (N, H, W) tensor of disparity.
            or
            depth: (N, H, W) tensor of depth.
        Returns:
            objseg_masks: list of MaskData
        """
        all_masks = data[f'{self.in_prefix}masks']
        objseg_masks = [None for _ in range(len(all_masks))]
        for i in range(len(all_masks)):            
            if f'{self.in_prefix}disparity' in data:
                disps = data[f'{self.in_prefix}disparity'][i]
            else:
                disps = 1 / data[f'{self.in_prefix}depth'][i]
            objseg_masks[i] = self._postprocess_depth_occlusion(all_masks[i], disps)
            objseg_masks[i] = self._nms(objseg_masks[i])
        data[f'{self.out_prefix}masks'] = objseg_masks

        return data

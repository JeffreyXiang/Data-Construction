# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
import cv2
from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from .automatic_mask_generator import SamAutomaticMaskGenerator


class SamAutomaticObjectSegmenter(SamAutomaticMaskGenerator):
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        depth_occlusion_dilate_radius: int = 3,
        depth_occlusion_thresh: float = 0.01,
        output_mode: str = "binary_mask",
    ) -> None:
        super().__init__(
            model,
            points_per_side,
            points_per_batch,
            pred_iou_thresh,
            stability_score_thresh,
            stability_score_offset,
            box_nms_thresh,
            crop_n_layers,
            crop_nms_thresh,
            crop_overlap_ratio,
            crop_n_points_downscale_factor,
            point_grids,
            min_mask_region_area,
            output_mode,
        )
        self.depth_occlusion_dilate_radius = depth_occlusion_dilate_radius
        self.depth_occlusion_thresh = depth_occlusion_thresh

    
    def _postprocess_depth_occlusion(
        self, masks: MaskData, depth: torch.Tensor, dilate_radius: int = 3, thresh: float = 0.01
    ) -> MaskData:
        """
        Removes masks that are occluded by its neighbors in depth.
        """
        if len(masks["boxes"]) == 0:
            return masks
        
        # Filter masks by depth occlusion
        depth = depth.cpu().numpy()
        depth = depth / np.max(depth)
        keep = []
        for idx in range(len(masks["rles"])):
            mask = rle_to_mask(masks["rles"][idx])

            # filter out masks whose neighbors are closer to the camera
            mask_dilated = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_radius, dilate_radius)))
            depth_masked = np.where(mask_dilated, 0, depth)
            depth_masked_dilated = cv2.dilate(depth_masked, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_radius + 1, dilate_radius + 1)))

            # filter out masks that are too close to the edge
            depth_masked_dilated[0, :] = 0
            depth_masked_dilated[-1, :] = 0
            depth_masked_dilated[:, 0] = 0
            depth_masked_dilated[:, -1] = 0
            
            max_diff = np.max(depth_masked_dilated - depth)
            if max_diff < thresh:
                keep.append(idx)

        # Filter masks
        masks.filter(keep)

        return masks


    @torch.no_grad()
    def generate(self, image: torch.Tensor, depth: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
            image (torch.Tensor): The image to generate masks for. Should be a
                tensor of shape (3, H, W) and dtype float32.
            depth (torch.Tensor): The depth map for the image. Should be a tensor
                of shape (H, W) and dtype float32.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """
        np_image = np.clip(image.cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # Generate masks
        mask_data = self._generate_masks(np_image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Filter masks by depth occlusion
        masks = self._postprocess_depth_occlusion(masks, depth, self.depth_occlusion_dilate_radius, self.depth_occlusion_thresh)

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

  
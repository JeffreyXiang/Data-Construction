import numpy as np
import torch
from typing import *
import time
import utils3d
import pipelines
from datasets.sa1b import SA_1B
from utils.basic_utils import *


if __name__ == "__main__":
    from torchvision import utils
    
    rastctx = utils3d.torch.RastContext(backend='gl')
    dataset = SA_1B("../SA-1B", return_annotation=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    pipe = pipelines.Compose([
        pipelines.nodes.DisparityPrediction(),
        pipelines.nodes.AnnotationToMask(),
        pipelines.nodes.ObjectMaskFiltering(min_mask_region_size=512),
    ])
    images = []
    smoothen_images = []
    whole_images = []
    whole_depths = []
    NUM = 25
    count = 0
    for i, batch in enumerate(dataloader):
        if count >= NUM: break
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        data = pipe(batch, timing=False)
        for image, depth, masks in zip(data['image'], data['disparity'], data['objseg_masks']):
            for mask in masks:
                alpha = mask['segmentation']
                img = np.concatenate([image.permute(1, 2, 0).cpu().numpy(), alpha[..., None]], axis=2)
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                img = img[mask['bbox'][1]:mask['bbox'][1] + mask['bbox'][3], mask['bbox'][0]:mask['bbox'][0] + mask['bbox'][2]]
                images.append(pad_image(img, 512, 'max_fit'))
                # smoothen_alpha = soften_mask(alpha)
                # img = np.concatenate([image.permute(1, 2, 0).cpu().numpy(), smoothen_alpha[..., None]], axis=2)
                # img = np.clip(img * 255, 0, 255).astype(np.uint8)
                # img = img[mask['bbox'][1]:mask['bbox'][1] + mask['bbox'][3], mask['bbox'][0]:mask['bbox'][0] + mask['bbox'][2]]
                # smoothen_images.append(pad_image(img, 512, 'max_fit'))
                img = image.permute(1, 2, 0).cpu().numpy()
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                whole_images.append(pad_image(img, 512, 'max_fit'))
                img = depth.cpu().numpy()[..., None].repeat(3, axis=-1)
                img = (img - img.min()) / (img.max() - img.min())
                img = (img * 255).astype(np.uint8)
                whole_depths.append(pad_image(img, 512, 'max_fit'))
                count += 1
                
    images = torch.tensor(np.stack(images)).permute(0, 3, 1, 2) / 255
    utils.save_image(images, "construction_pipe_object.png", nrow=10, normalize=True, value_range=(0, 1))
    # smoothen_images = torch.tensor(np.stack(smoothen_images)).permute(0, 3, 1, 2) / 255
    # utils.save_image(smoothen_images, "construction_pipe_object_smoothen.png", nrow=10, normalize=True, value_range=(0, 1))
    whole_images = torch.tensor(np.stack(whole_images)).permute(0, 3, 1, 2) / 255
    utils.save_image(whole_images, "construction_pipe_whole.png", nrow=10, normalize=True, value_range=(0, 1))
    whole_depths = torch.tensor(np.stack(whole_depths)).permute(0, 3, 1, 2) / 255
    utils.save_image(whole_depths, "construction_pipe_whole_depth.png", nrow=10, normalize=True)
    

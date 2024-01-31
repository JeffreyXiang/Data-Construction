import numpy as np
import torch
from typing import *
import utils3d
from data_construction import pipelines
from data_construction.datasets.sa1b import SA_1B


if __name__ == "__main__":
    from torchvision import utils
    
    rastctx = utils3d.torch.RastContext(backend='gl')
    dataset = SA_1B("../SA-1B", image_size=512, crop=True, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    pipe = pipelines.Compose([
        pipelines.nodes.BboxVisualization(),
        pipelines.nodes.DepthPrediction(),
        pipelines.nodes.FovSetting(),
        pipelines.nodes.RandomWarping(ctx=rastctx),
        pipelines.nodes.Inpainting(),
        pipelines.nodes.DepthPrediction(in_prefix="inpainted_warped_"),
        pipelines.nodes.DepthAlignment(),
        pipelines.nodes.BackWarping(ctx=rastctx),
    ])
    image = []
    bbox_image = []
    warp_image = []
    warp_dr = []
    inpaint_image = []
    backwarp_image = []
    count = 0
    NUM = 100
    for i, batch in enumerate(dataloader):
        if count >= NUM: break
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        data = pipe(batch)
        image.append(data['image'])
        bbox_image.append(data['bbox_image'])
        warp_image.append(data['warped_image'] * data['warped_mask'][:, None])
        warp_dr.append(data['warped_dr'][:, 4])
        inpaint_image.append(data['inpainted_warped_image'])
        backwarp_image.append(data['backwarped_inpainted_warped_image'] * data['backwarped_inpainted_warped_mask'][:, None])
        count += len(data['image'])
    utils.save_image(torch.cat(image), "construction_pipe.jpg", nrow=int(np.sqrt(NUM)), normalize=True, range=(0, 1))
    utils.save_image(torch.cat(bbox_image), "construction_pipe_bbox.jpg", nrow=int(np.sqrt(NUM)), normalize=True, range=(0, 1))
    utils.save_image(torch.cat(warp_image), "construction_pipe_warp.jpg", nrow=int(np.sqrt(NUM)), normalize=True, range=(0, 1))
    utils.save_image(torch.cat(warp_dr)[:, None], "construction_pipe_warp_dr.jpg", nrow=int(np.sqrt(NUM)), range=(0, 1))
    utils.save_image(torch.cat(inpaint_image), "construction_pipe_inpaint.jpg", nrow=int(np.sqrt(NUM)), normalize=True, range=(0, 1))
    utils.save_image(torch.cat(backwarp_image), "construction_pipe_backwarp.jpg", nrow=int(np.sqrt(NUM)), normalize=True, range=(0, 1))
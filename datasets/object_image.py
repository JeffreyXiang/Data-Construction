import os
import numpy as np
import torch
from torchvision import transforms
import PIL.Image as Image
from typing import *
import utils3d

from .monodepth_utils import *
from .construction_pipe import DepthWarpPipe


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
            'image': image,
        }


if __name__ == "__main__":
    from torchvision import utils
    
    rastctx = utils3d.torch.RastContext(backend='cuda')
    dataset = ObjectImageDataset("/home/sichengxu/jianfeng/datasets/SDIP_dog", image_size=512, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)
    pipe = DepthWarpPipe([
        DepthWarpPipe.DepthPrediction(),
        DepthWarpPipe.BackgroundRemoval(),
        DepthWarpPipe.ForegroundPoint(),
        DepthWarpPipe.PointVisualization(),
        DepthWarpPipe.Resize(size=128),
        DepthWarpPipe.FovSetting(),
        DepthWarpPipe.RandomWarping(ctx=rastctx),
        DepthWarpPipe.IvidInpainting(),
        DepthWarpPipe.DepthPrediction(in_prefix='inpainted_warped_'),
        DepthWarpPipe.DepthAlignment(),
        DepthWarpPipe.BackWarping(ctx=rastctx),
    ])
    image = []
    rmbg_image = []
    point_image = []
    warp_image = []
    warp_dr = []
    inpaint_image = []
    backwarp_image = []
    flow_image = []
    count = 0
    NUM = 100
    for i, batch in enumerate(dataloader):
        if count >= NUM: break
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        data = pipe(batch)
        print(list(data.keys()))
        image.append(data['image'])
        rmbg_image.append(data['rmbg_image'])
        point_image.append(data['point_image'])
        warp_image.append(data['warped_image'] * data['warped_mask'][:, None])
        warp_dr.append(data['warped_dr'][:, 4])
        inpaint_image.append(data['inpainted_warped_image'])
        backwarp_image.append(data['backwarped_inpainted_warped_image'] * data['backwarped_inpainted_warped_mask'][:, None])
        flow_image.append(data['backwarped_inpainted_warped_flow'])
        count += len(data['image'])
    utils.save_image(torch.cat(image), "construction_pipe.jpg", nrow=int(np.sqrt(NUM)), normalize=True, value_range=(0, 1))
    utils.save_image(torch.cat(rmbg_image), "construction_pipe_rmbg.jpg", nrow=int(np.sqrt(NUM)), normalize=True, value_range=(0, 1))
    utils.save_image(torch.cat(point_image), "construction_pipe_point.jpg", nrow=int(np.sqrt(NUM)), normalize=True, value_range=(0, 1))
    utils.save_image(torch.cat(warp_image), "construction_pipe_warp.jpg", nrow=int(np.sqrt(NUM)), normalize=True, value_range=(0, 1))
    utils.save_image(torch.cat(warp_dr)[:, None], "construction_pipe_warp_dr.jpg", nrow=int(np.sqrt(NUM)))
    utils.save_image(torch.cat(inpaint_image), "construction_pipe_inpaint.jpg", nrow=int(np.sqrt(NUM)), normalize=True, value_range=(0, 1))
    utils.save_image(torch.cat(backwarp_image), "construction_pipe_backwarp.jpg", nrow=int(np.sqrt(NUM)), normalize=True, value_range=(0, 1))
    utils.save_image(torch.cat(flow_image), "construction_pipe_flow.jpg", nrow=int(np.sqrt(NUM)), normalize=True, value_range=(-0.1, 0.1))
    
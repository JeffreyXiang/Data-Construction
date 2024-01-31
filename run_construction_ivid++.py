import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import hashlib
import importlib
import argparse
import numpy as np
from tqdm import tqdm
import io
import cv2
import imageio
import OpenEXR as exr
import Imath as exr_types
import pickle
import threading
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import utils

import utils3d
from data_construction import pipelines
from data_construction.utils.basic_utils import pack_image


def setup_dist(rank, local_rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def save_data(data, count, dir):
    num = len(data['image'])
    for i in range(num):
        # Check NaN
        nan_keys = [k for k, v in data.items() if isinstance(v, torch.Tensor) and torch.isnan(v[i]).any()]
        if len(nan_keys) > 0:
            print(f"\033[93mWarning: NaN detected in data {count + i}, entries: {', '.join(nan_keys)}\033[0m")

        # Save data
        image = np.clip(data['image'][i].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        rmgb_image = np.clip(data['rmbg_image'][i].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        point_image = np.clip(data['point_image'][i].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        warped_image = np.clip(data['warped_image'][i].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        warped_mask = np.clip(data['warped_mask'][i].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        inpainted_warped_image = np.clip(data['inpainted_warped_image'][i].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        backwarped_inpainted_warped_image = np.clip(data['backwarped_inpainted_warped_image'][i].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        backwarped_inpainted_warped_mask = np.clip(data['backwarped_inpainted_warped_mask'][i].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        backwarped_inpainted_warped_dr = data['backwarped_inpainted_warped_dr'][i].cpu().numpy().astype(np.float16)
        backwarped_inpainted_warped_dr = backwarped_inpainted_warped_dr[4] * (backwarped_inpainted_warped_dr[5] < 0)
        depth = data['depth'][i].cpu().numpy().astype(np.float16)
        inpainted_warped_depth = data['inpainted_warped_depth'][i].cpu().numpy().astype(np.float16)
        transform = data['transform'][i].cpu().numpy()
        fov = data['fov']
        flow = data['backwarped_inpainted_warped_flow'][i].cpu().numpy().transpose(1, 2, 0).astype(np.float16)

        out = {
            'image': pack_image(image),
            'inpainted_warped_image': pack_image(np.concatenate([inpainted_warped_image, warped_mask[..., None]], axis=-1)),
            'backwarped_inpainted_warped_image': pack_image(np.concatenate([backwarped_inpainted_warped_image, backwarped_inpainted_warped_mask[..., None]], axis=-1)),
            'depth': pack_image(depth),
            'inpainted_warped_depth': pack_image(inpainted_warped_depth),
            'backwarped_inpainted_warped_dr': pack_image(backwarped_inpainted_warped_dr),
            'transform': transform,
            'fov': fov,
            'flow': pack_image(flow),
        }
        with open(os.path.join(dir, 'data', f'{count}.pkl'), 'wb') as f:
            pickle.dump(out, f)

        # Save visualization
        rmgb_image = cv2.resize(rmgb_image, (128, 128), interpolation=cv2.INTER_AREA)
        point_image = cv2.resize(point_image, (128, 128), interpolation=cv2.INTER_AREA)
        warped_image = warped_image * (warped_mask[..., None] > 127)
        backwarped_inpainted_warped_image = backwarped_inpainted_warped_image * (backwarped_inpainted_warped_mask[..., None] > 127)
        backwarped_inpainted_warped_dr = np.clip(backwarped_inpainted_warped_dr * 255, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=-1)
        flow_vis = (utils.flow_to_image(data['backwarped_inpainted_warped_flow'][i, :2]) * data['backwarped_inpainted_warped_flow'][i, 2:]).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        flow_vis *= (backwarped_inpainted_warped_mask[..., None] > 127)
        vis = np.concatenate([image, rmgb_image, point_image, warped_image, inpainted_warped_image, backwarped_inpainted_warped_image, backwarped_inpainted_warped_dr, flow_vis], axis=1)
        imageio.imwrite(os.path.join(dir, 'vis', f'{count}.jpg'), vis)

        count += 1


def main(local_rank, cfg):
    setup_dist(local_rank, local_rank, cfg.num_gpus, cfg.master_addr, cfg.master_port)

    rastctx = utils3d.torch.RastContext(backend='cuda')
    dataset = getattr(importlib.import_module(cfg.dataset.split('.')[0]), cfg.dataset.split('.')[1])(cfg.data_dir, image_size=cfg.image_size, normalize=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=4,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=cfg.num_gpus,
            rank=local_rank,
            shuffle=False,
        )
    )
    pipe = pipelines.Compose([
        pipelines.nodes.DepthPrediction(),
        pipelines.nodes.BackgroundRemoval(),
        pipelines.nodes.ForegroundPoint(),
        pipelines.nodes.PointVisualization(),
        pipelines.nodes.Resize(size=128),
        pipelines.nodes.FovSetting(),
        pipelines.nodes.RandomWarping(ctx=rastctx),
        pipelines.nodes.IvidInpainting(),
        pipelines.nodes.DepthPrediction(in_prefix='inpainted_warped_'),
        pipelines.nodes.DepthAlignment(),
        pipelines.nodes.BackWarping(ctx=rastctx),
    ])

    threads = []
    count = 8 * local_rank
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Rank {local_rank}", position=local_rank):
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        data = pipe(batch)
        thread = threading.Thread(target=save_data, args=(data, count, cfg.output_dir))
        thread.start()
        threads.append(thread)
        count += len(data['image']) * cfg.num_gpus
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='object_image.ObjectImageDataset', help='Dataset name')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--data_dir', type=str, default='/home/sichengxu/jianfeng/datasets/SDIP_dog', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./datasets/SDIP_dogs', help='Output directory')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345', help='Port for distributed training')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    cfg = parser.parse_args()
    cfg.num_gpus = torch.cuda.device_count() if cfg.num_gpus == -1 else cfg.num_gpus
    os.makedirs(os.path.join(os.path.dirname(__file__), 'tmp'), exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'vis'), exist_ok=True)

    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
    
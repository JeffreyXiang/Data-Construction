import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import argparse
import numpy as np
from tqdm import tqdm
import imageio
import pickle
import threading
import torch
import torch.multiprocessing as mp
import utils3d
from data_construction import pipelines, datasets
from data_construction.utils.basic_utils import pack_image
from data_construction.datasets.wrapper import Part


def save_data(data, cfg):
    num = len(data['filename'])
    for i in range(num):
        # Save data
        image = np.clip(data['image'][i*cfg.nviews].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        depth = data['depth'][i*cfg.nviews].cpu().numpy().astype(np.float16)
        inpainted_warped_image = [np.clip(data['inpainted_warped_image'][j].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8) for j in range(i*cfg.nviews, (i+1)*cfg.nviews)]
        inpainted_warped_depth = [data['inpainted_warped_depth'][j].cpu().numpy().astype(np.float16) for j in range(i*cfg.nviews, (i+1)*cfg.nviews)]
        transform = [data['transform'][j].cpu().numpy() for j in range(i*cfg.nviews, (i+1)*cfg.nviews)]
        fov = [data['fov'][j].item() for j in range(i*cfg.nviews, (i+1)*cfg.nviews)]

        out = {
            'image': pack_image(image),
            'depth': pack_image(depth),
            'inpainted_warped_image': [pack_image(img) for img in inpainted_warped_image],
            'inpainted_warped_depth': [pack_image(d) for d in inpainted_warped_depth],
            'transform': transform,
            'fov': fov,
        }

        with open(os.path.join(cfg.output_dir, 'data', f'{data["filename"][i]}.pkl'), 'wb') as f:
            pickle.dump(out, f)

        # Save visualization
        vis = np.concatenate([image] + inpainted_warped_image, axis=1)
        imageio.imwrite(os.path.join(cfg.output_dir, 'vis', f'{data["filename"][i]}.jpg'), vis)


def main(local_rank, cfg):
    torch.cuda.set_device(local_rank % cfg.num_gpus)
    rastctx = utils3d.torch.RastContext(backend='cuda')
    dataset = getattr(datasets, cfg.dataset)(cfg.data_dir, image_size=cfg.image_size, normalize=False)
    dataset = Part(dataset, cfg.total_parts, cfg.part)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=cfg.num_procs,
            rank=local_rank,
            shuffle=False,
        )
    )
    pipe = pipelines.Compose([
        pipelines.nodes.DepthPrediction(),
        pipelines.nodes.BackgroundRemoval(),
        pipelines.nodes.Replicate(cfg.nviews),
        pipelines.nodes.ForegroundPoint(),
        pipelines.nodes.PointVisualization(),
        pipelines.nodes.Resize(size=cfg.resize),
        pipelines.nodes.FovRandomSetting(),
        pipelines.nodes.RandomWarping(ctx=rastctx),
        pipelines.nodes.IvidInpainting(),
        pipelines.nodes.DepthPrediction(in_prefix='inpainted_warped_'),
        pipelines.nodes.DepthAlignment(),
    ])

    threads = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Rank {local_rank}", position=local_rank):
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        data = pipe(batch)
        thread = threading.Thread(target=save_data, args=(data, cfg))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ImageDataset', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='../SDIP_dog/SDIP_dog', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./datasets/SDIP_dogs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    # Data cfg
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--resize', type=int, default=128, help='Resize image')
    parser.add_argument('--nviews', type=int, default=16, help='Number of views')
    # Parallel
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--num_procs_per_gpu', type=int, default=1, help='Number of processes per GPU')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345', help='Port for distributed training')
    # Dataset split
    parser.add_argument('--total_parts', type=int, default=1, help='Total number of parts')
    parser.add_argument('--part', type=int, default=0, help='Part number')
    cfg = parser.parse_args()
    cfg.num_gpus = torch.cuda.device_count() if cfg.num_gpus == -1 else cfg.num_gpus
    cfg.num_procs = cfg.num_gpus * cfg.num_procs_per_gpu

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'vis'), exist_ok=True)

    mp.spawn(main, args=(cfg,), nprocs=cfg.num_procs, join=True)
    
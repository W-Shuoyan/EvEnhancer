import importlib
import numpy as np
import random

import torch
import cv2
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp

from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.data.transforms import augment, imresize_np
from basicsr.utils import get_root_logger, scandir, img2tensor
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]

# import all the dataset modules
_dataset_modules = [importlib.import_module(f'basicsr.data.{file_name}') for file_name in dataset_filenames]


def custom_collate_fn(batch):
    batch_size = len(batch)
    imgs = [b['imgs'] for b in batch]
    voxels = [b['voxels'] for b in batch]
    scale_min = batch[0]['scale_min']
    scale_max = batch[0]['scale_max']
    lq_size = batch[0]['lq_size']

    scale = random.uniform(scale_min, scale_max)
    times = np.linspace(0, 1, len(imgs[0]), dtype=float)

    h_lq, w_lq = lq_size
    h_gt = round(h_lq * scale)
    w_gt = round(w_lq * scale)
    
    all_voxels = []
    all_gts = []
    all_lqs = []
    for b_idx in range(batch_size):
        imgs_b = imgs[b_idx]
        h_img, w_img, _ = imgs_b[0].shape

        h0 = random.randint(0, h_img - h_gt)
        w0 = random.randint(0, w_img - w_gt)
        
        imgs_gt = []
        for img in imgs_b:
            img_gt = img[h0:h0 + h_gt, w0:w0 + w_gt, :]
            imgs_gt.append(img_gt)

        imgs_lq = []
        for img in (imgs_b[0], imgs_b[-1]):
            img_lq = img[h0:h0 + h_gt, w0:w0 + w_gt, :]
            img_lq = imresize_np(img_lq, 1 / scale, True)
            imgs_lq.append(img_lq)

        voxels_b = voxels[b_idx][h0:h0 + h_gt, w0:w0 + w_gt, :]
        voxels_b = [cv2.resize(voxels_b, (w_lq, h_lq), interpolation=cv2.INTER_CUBIC)]

        tensors = imgs_lq + imgs_gt + voxels_b
        tensors = augment(tensors, True, True)
        tensors = img2tensor(tensors)

        all_lqs.append(torch.stack(tensors[:len(imgs_lq)], dim=0))
        all_gts.append(torch.stack(tensors[len(imgs_lq):len(imgs_lq)+len(imgs_gt)], dim=0))
        
        voxels_b = torch.stack(tensors[len(imgs_lq)+len(imgs_gt):], dim=1)
        voxels_b = voxels_b.squeeze(1)
        all_voxel = []
        for i in range(voxels_b.shape[0]-1):
            all_voxel.append(voxels_b[i:i+2, :, :])
        voxels_b = torch.stack(all_voxel, dim=0)
        all_voxels.append(voxels_b)

    all_voxels = torch.stack(all_voxels, dim=0)
    all_gts = torch.stack(all_gts, dim=0)
    all_lqs = torch.stack(all_lqs, dim=0)

    return {'lq': all_lqs, 'gt': all_gts, 'voxel': all_voxels,
            'times': times,
            'scale': scale}
    


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    logger = get_root_logger()
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} is built.')
    return dataset


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
        dataloader_args['collate_fn'] = custom_collate_fn

    elif phase in ['val', 'test']:
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
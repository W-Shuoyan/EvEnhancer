from torch.utils import data as data
import os
import cv2
from pathlib import Path
import random
import numpy as np

import torch

from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from data.data_util import recursive_glob
from data.event_util import events_to_voxel_grid
from data.transforms import imresize_np


@DATASET_REGISTRY.register()
class AdobeGopro_val_Dataset(data.Dataset):
    def __init__(self, opt):
        super(AdobeGopro_val_Dataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.m = opt['num_skip_interpolation'] 
        self.n = opt['num_inter_interpolation'] 
        self.moments = opt['moment_events']
        assert self.m >= self.n, 'The number of skip interpolation must greater than or equal to the number of inter interpolation.'

        self.scale_min = opt['scale_min']
        self.scale_max = opt['scale_max']

        self.gt_setLength = self.m + 2
        self.lq_setLength = self.n + 2

        self.split = 'train' if opt['phase'] == 'train' else 'valid'

        train_video_list = os.listdir(os.path.join(self.dataroot, 'train'))
        test_video_list = os.listdir(
            os.path.join(self.dataroot, 'valid'))

        video_list = train_video_list if self.split == 'train' else test_video_list

        self.imageSeqsPath = []
        self.eventSeqsPath = []

        assert (self.m + 1) % (self.n + 1) == 0, 'If the selection of supervision frame is not random, then (number of skipped frames+1) needs to be divided by (number of interpolated frames+1)!'
        frame_idx = np.arange(0, self.m + 1, (self.m + 1) // (self.n + 1))
        frame_idx = np.append(frame_idx, self.m + 1)

        for video in video_list:
            frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'imgs'),
                                           suffix='.png'), key=lambda x: int(x.split('.')[0]))
            event_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, self.split, video, 'event_random'),
                                                 suffix='.npz'), key=lambda x: int(x.split('.')[0]))
            n_sets = (len(frames) - 1) // (self.gt_setLength - 1)

            videoInputs = [[frames[(self.gt_setLength - 1) * i + idx] for idx in frame_idx] for i in range(n_sets)]
            videoInputs = [[os.path.join(self.dataroot, self.split, video, 'imgs', f)
                            for f in group] for group in videoInputs]
            self.imageSeqsPath.extend(videoInputs)
       
            eventInputs = [event_frames[(self.gt_setLength - 1) * i:(self.gt_setLength - 1) *
                                        i + self.gt_setLength - 1] for i in range(n_sets)]
            eventInputs = [[os.path.join(self.dataroot, self.split, video, 'event_random', f)
                            for f in group] for group in eventInputs]
            self.eventSeqsPath.extend(eventInputs)

        self.file_client = None
        self.io_backend_opt = opt['io_backend']


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        img_paths = self.imageSeqsPath[index]
        event_paths = self.eventSeqsPath[index]

        imgs = []
        for img_path in img_paths:
            img = self.file_client.get(img_path)
            img = imfrombytes(img, float32=True)
            imgs.append(img)

        h_img, w_img, _ = imgs[0].shape
        events = [np.load(event_path) for event_path in event_paths]
        events = np.concatenate([
            np.column_stack((event['t'], event['x'], event['y'], event['p'])).astype(np.float32)
            for event in events], axis=0)
        voxels = events_to_voxel_grid(events, num_bins=self.moments+1, width=w_img, height=h_img,
                                        return_format='HWC')

        img_path = img_paths[0]
        seq = img_path.split(f'{self.split}/')[1].split('/')[0]
        origin_index = os.path.basename(img_path).split('.')[0]

        scale = random.uniform(self.scale_min, self.scale_max)
        times = np.linspace(0, 1, len(imgs), dtype=float)

        h_lq, w_lq = self.opt['lq_size']
        h_gt = round(h_lq * scale)
        if h_gt > h_img:
            h_gt = h_img
        w_gt = round(w_lq * scale)   
        if w_gt > w_img:
            w_gt = w_img     
        h0 = h_img - h_gt
        w0 = w_img - w_gt
        
        imgs_gt = []
        for img in imgs:
            img_gt = img[h0:h0 + h_gt, w0:w0 + w_gt, :]
            img_gt = img2tensor(img_gt)
            imgs_gt.append(img_gt)

        imgs_lq = []
        for img in (imgs[0], imgs[-1]):
            img_lq = img[h0:h0 + h_gt, w0:w0 + w_gt, :]
            img_lq = imresize_np(img_lq, 1 / scale, True)
            img_lq = img2tensor(img_lq)
            imgs_lq.append(img_lq)

        voxels = voxels[h0:h0 + h_gt, w0:w0 + w_gt, :]
        voxels = cv2.resize(voxels, (w_lq, h_lq), interpolation=cv2.INTER_CUBIC)
        voxels = img2tensor(voxels)
        all_voxel = []
        for i in range(voxels.shape[0]-1):
            sub_voxel = voxels[i:i+2, :, :]
            all_voxel.append(sub_voxel)
        voxels = torch.stack(all_voxel, dim=0)
        
        imgs_lq = torch.stack(imgs_lq, dim=0)
        imgs_gt = torch.stack(imgs_gt, dim=0)

        return {'lq': imgs_lq, 
                'gt': imgs_gt, 
                'voxel':  voxels,
                'scale': scale,
                'times': times,
                'seq': seq,
                'origin_index': origin_index,
                }

    def __len__(self):
        return len(self.imageSeqsPath)

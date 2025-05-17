from torch.utils import data as data
import os
import cv2
from pathlib import Path
import random
import numpy as np

from basicsr.utils import FileClient, imfrombytes
from basicsr.utils.registry import DATASET_REGISTRY

from data.data_util import recursive_glob
from data.event_util import events_to_voxel_grid


@DATASET_REGISTRY.register()
class Adobe240_train_Dataset(data.Dataset):
    def __init__(self, opt):
        super(Adobe240_train_Dataset, self).__init__()
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

        self.split = 'train' if opt['phase'] == 'train' else 'test'

        train_video_list = os.listdir(os.path.join(self.dataroot, 'train'))
        test_video_list = os.listdir(
            os.path.join(self.dataroot, 'test'))  

        video_list = train_video_list if self.split == 'train' else test_video_list

        self.imageSeqsPath = [] 
        self.eventSeqsPath = []

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

        return {'imgs': imgs, 
                'voxels': voxels,
                'scale_min': self.scale_min,
                'scale_max': self.scale_max,
                'lq_size': self.opt['lq_size'],
                'moments': self.moments
                }

    def __len__(self):
        return len(self.imageSeqsPath)

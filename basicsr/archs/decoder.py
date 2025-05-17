import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .function import make_coord, get_coords, get_cells, get_idxlist
from .module_util import PositionEncoder3d


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, act='gelu'):
        super().__init__()

        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU(True)
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        else:
            assert False, f'activation {act} is not supported'

        layers = []
        lastv = in_dim

        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden

        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class LIVT_Decoder(nn.Module):
    def __init__(
        self,
        in_dim=64,
        base_dim=16,
        head=8,
        r=3,
        r_t=2
    ):
        super().__init__()
        self.in_dim = in_dim
        self.dim = base_dim
        self.head = head
        self.r = r
        self.r_t = r_t

        self.conv_ch = nn.Conv3d(
            self.in_dim, self.dim, kernel_size=3, padding=1)

        self.conv_vs = nn.Conv3d(self.dim, self.dim, kernel_size=3, padding=1)
        
        self.conv_qs = nn.Conv3d(self.dim, self.dim, kernel_size=3, padding=1)
        
        self.conv_ks = nn.Conv3d(self.dim, self.dim, kernel_size=3, padding=1)

        self.pb_encoder = PositionEncoder3d(posenc_scale=10, enc_dims=64, gamma=1)

        self.r_area = (2 * self.r + 1)**2
        self.r_volume = self.r_area * (2 * self.r_t + 1)

        imnet_in_dim = self.dim * self.r_volume + self.dim + 2

        self.mlp = MLP(in_dim=imnet_in_dim, out_dim=3, hidden_list=[256, 256, 256, 256], act='gelu')

    def forward(self, feat_img, scale, times):
        feat_img_shape = feat_img.shape
        coords = get_coords(feat_img_shape, scale)

        sr_image_list = []

        # Feature generation
        feat = self.conv_ch(feat_img)  # b, c, t, h, w
        del feat_img
        torch.cuda.empty_cache()
        bs, fc, ft, fh, fw = feat.shape
        times = times * (ft - 1)

        # Query RGB
        coord_lr = make_coord((fh, fw), flatten=False).cuda()
        coord_lr = coord_lr.permute(2, 0, 1).unsqueeze(
            0).repeat(bs, 1, 1, 1)  # b, 2, h, w

        hr_coord = coords.clone()
        hr_coord = hr_coord.reshape(bs, -1, 2)
        hr_coord = hr_coord.unsqueeze(2)  # b, q, 1, 2
        q_sample = hr_coord.shape[1]

        # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
        sample_coord_k = F.grid_sample(
            coord_lr, hr_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)

        del coord_lr
        torch.cuda.empty_cache()

        # field radius (global: [-1, 1])
        rh = 2 / fh
        rw = 2 / fw
        r = self.r
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * rh
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * rw
        # 1, 1, r_area, 2
        delta = torch.stack(torch.meshgrid(
            dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)

        # b, q, 1, 2 -> b, q, 1, r_area, 2
        sample_coord_k = sample_coord_k + delta
        del delta
        torch.cuda.empty_cache()


        feat_q = self.conv_qs(feat)
        feat_k = self.conv_ks(feat)
        feat_v = self.conv_vs(feat)

        # b, 2 -> b, q, 2
        rel_cell = get_cells(feat_img_shape, scale).cuda()
        rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)
        rel_cell[..., 0] *= fh
        rel_cell[..., 1] *= fw

        for i in range(len(times)):
            center_time = round(times[i])
            rel_time = (times[i] - center_time) * 2 / (2 * self.r_t + 1)
            sample_idx = get_idxlist(center_time, self.r_t, 0, feat_img_shape[2]-1)
            hr_coord3d = torch.cat((torch.ones_like(
                hr_coord[..., :1])*rel_time, hr_coord), dim=-1).float().unsqueeze(3)
            # Q - b, c, t, h, w -> b, c, q, 1, 1 -> b, q, 1, 1, c -> b, q, 1, h, c/h -> b, q, h, 1, c/h
            sample_feat_q = F.grid_sample(
                feat_q[:, :, sample_idx, ...], hr_coord3d.flip(-1), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 4, 1)
            sample_feat_q = sample_feat_q.reshape(
                bs, q_sample, 1, self.head, self.dim // self.head
            ).permute(0, 1, 3, 2, 4)

            # b, q, r_area, 3
            rel_coord_0 = hr_coord - sample_coord_k
           

            feat_in = []
            # Unfold along the temporal dimension
            for dt in range(2 * self.r_t + 1):
                time_coord = rel_time - (dt - self.r_t) * 2 / (2 * self.r_t + 1)
                rel_coord = torch.cat((torch.ones_like(
                rel_coord_0[..., :1])*time_coord, rel_coord_0), dim=-1).float()
                rel_coord[..., 0] *= 2 * self.r_t + 1
                rel_coord[..., 1] *= fh
                rel_coord[..., 2] *= fw

                # b, q, r_area, h
                _, pb = self.pb_encoder(rel_coord)
                del rel_coord
                torch.cuda.empty_cache()
                # K - b, c, h, w -> b, c, q, r_area -> b, q, r_area, c -> b, q, r_area, h, c/h -> b, q, h, c/h, r_area
                sample_feat_k = F.grid_sample(
                    feat_k[:, :, sample_idx, ...][:, :, dt, ...], sample_coord_k.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)
                sample_feat_k = sample_feat_k.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                ).permute(0, 1, 3, 4, 2)

                # b, q, h, 1, r_area -> b, q, r_area, h
                attn = torch.matmul(sample_feat_q, sample_feat_k).reshape(
                    bs, q_sample, self.head, self.r_area
                ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)
                del sample_feat_k
                torch.cuda.empty_cache()
                attn = F.softmax(torch.add(attn, pb), dim=-2)
                attn = attn.reshape(
                    bs, q_sample, self.r_area, self.head, 1)

                # V - b, c, h, w -> b, c, q, r_area -> b, q, r_area, c
                sample_feat_v = F.grid_sample(
                    feat_v[:, :, sample_idx, ...][:, :, dt, ...], sample_coord_k.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)

                sample_feat_v = sample_feat_v.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                )
                attn = torch.mul(sample_feat_v, attn).reshape(
                    bs, q_sample, -1)
                del sample_feat_v
                torch.cuda.empty_cache()
                feat_in.append(attn)
                del attn
                torch.cuda.empty_cache()
            feat_in = torch.cat(feat_in, dim=-1)
            feat_back = F.grid_sample(
                feat_q[:, :, sample_idx, ...], hr_coord3d.flip(-1), mode='bilinear', align_corners=False
            ).permute(0, 2, 1, 3, 4).reshape(bs, q_sample, fc)
            feat_in = torch.cat([feat_in, feat_back, rel_cell], dim=-1)
            del feat_back, hr_coord3d
            torch.cuda.empty_cache()
            pred = self.mlp(feat_in).permute(0, 2, 1).reshape(
                bs, 3, coords.shape[1], coords.shape[2])  # b, c, h, w
            del feat_in
            torch.cuda.empty_cache()

            sr_image_list.append(pred)
            del pred
            torch.cuda.empty_cache()

        return sr_image_list

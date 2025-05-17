import torch
import copy

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def pair(t):
    return t if isinstance(t, tuple) else (t, t)



def sample_system_scale(scale_factor, scale_base):
    scale_it = []
    s = copy.copy(scale_factor)

    if s <= scale_base:
        scale_it.append(s)
    else:
        scale_it.append(scale_base)
        s = s / scale_base

        while s > 1:
            if s >= scale_base:
                scale_it.append(scale_base)
            else:
                scale_it.append(s)
            s = s / scale_base

    return scale_it


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def get_coords(feat_shape, scale):
    batch_size, _, _, lr_h, lr_w = feat_shape
    img_h = round(lr_h * scale)
    img_w = round(lr_w * scale)
    coords = make_coord((img_h, img_w), flatten=False).unsqueeze(
        0).repeat(batch_size, 1, 1, 1).cuda()
    return coords

def get_cells(feat_shape, scale):
    batch_size, _, _, lr_h, lr_w = feat_shape
    img_h = round(lr_h * scale)
    img_w = round(lr_w * scale)
    cell = torch.ones(2)
    cell[0] *= 2. / img_h
    cell[1] *= 2. / img_w
    cell = cell.unsqueeze(0).repeat(batch_size, 1)
    return cell

def get_idxlist(idx, r_t, min, max):
    idxlist = []
    for i in range(-r_t, r_t + 1):
        i += idx
        if i < min:
            i = min
        elif i > max:
            i = max
        idxlist.append(i)
    return idxlist


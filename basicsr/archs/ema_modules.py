import torch
from torch import nn

from .DCNv2.dcn_v2 import DCN_sep
from .arch_util import make_layer, ResidualBlockNoBN

class EMB(nn.Module):
    def __init__(self, nf):
        super(EMB, self).__init__()
        self.event_process = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.img_process = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, img, event):
        img = self.img_process(img)
        event = self.event_process(event)
        return img * event


class EPCD_Align(nn.Module):
    ''' Event-guided Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels, modified from TMNet
    '''
    def __init__(self, nf):
        super(EPCD_Align, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=8)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=8)
        self.L2_fea_conv_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=8)
        self.L1_fea_conv_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=8)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=8)
        self.L2_fea_conv_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=8)
        self.L1_fea_conv_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)

        self.EMB_A_l1 = EMB(nf)
        self.EMB_B_l1 = EMB(nf)
        self.EMB_A_l2 = EMB(nf)
        self.EMB_B_l2 = EMB(nf)
        self.EMB_A_l3 = EMB(nf)
        self.EMB_B_l3 = EMB(nf)

    def forward(self, fea1, fea2, event1_t, event2_t):
        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(
            L3_offset)) + self.EMB_A_l3(L3_offset, event1_t[2])
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        B, C, L2_H, L2_W = fea1[1].size()
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = nn.functional.interpolate(
            L3_offset, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(
            torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(
            L2_offset)) + self.EMB_A_l2(L2_offset, event1_t[1])
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = nn.functional.interpolate(
            L3_fea, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(
            torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        B, C, L1_H, L1_W = fea1[0].size()
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = nn.functional.interpolate(
            L2_offset, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(
            torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(
            L1_offset)) + self.EMB_A_l1(L1_offset, event1_t[0])
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = nn.functional.interpolate(
            L2_fea, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(
            L3_offset)) + self.EMB_B_l3(L3_offset, event2_t[2])
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = nn.functional.interpolate(
            L3_offset, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(
            torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(
            L2_offset)) + self.EMB_B_l2(L2_offset, event2_t[1])
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = nn.functional.interpolate(
            L3_fea, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(
            torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = nn.functional.interpolate(
            L2_offset, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(
            torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(
            L1_offset)) + self.EMB_B_l1(L1_offset, event2_t[0])
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = nn.functional.interpolate(
            L2_fea, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        y = torch.cat(y, dim=1)
        y = self.fusion(y)
        return y


class EventSubHead(nn.Module):
    def __init__(self, front_RBs, inf, nf):
        super(EventSubHead, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.feature_extraction = make_layer(ResidualBlockNoBN, front_RBs, num_feat=nf)
        self.fea_E1_conv1 = nn.Conv2d(inf, nf, 3, padding=1)
        self.fea_E2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_E2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_E3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_E3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, event):
        event_b, _, event_h, event_w = event.shape
        # E1_1
        E1_fea = self.lrelu(self.fea_E1_conv1(event))
        E1_fea = self.feature_extraction(E1_fea)
        # E2_1
        E2_fea = self.lrelu(self.fea_E2_conv1(E1_fea))
        E2_fea = self.lrelu(self.fea_E2_conv2(E2_fea))
        # E3_1
        E3_fea = self.lrelu(self.fea_E3_conv1(E2_fea))
        E3_fea = self.lrelu(self.fea_E3_conv2(E3_fea))
        E1_fea = E1_fea.view(event_b, -1, event_h, event_w)
        E2_fea = E2_fea.view(event_b, -1, E2_fea.shape[2], E2_fea.shape[3])
        E3_fea = E3_fea.view(event_b, -1, E3_fea.shape[2], E3_fea.shape[3])
        return [
            E1_fea.clone(), E2_fea.clone(), E3_fea.clone()
        ]


class EMA_Encoder(nn.Module):
    def __init__(self, front_RBs, inf, nf):
        super(EMA_Encoder, self).__init__()
        self.epcd_align = EPCD_Align(nf=nf)
        self.EventSubHead_1 = EventSubHead(front_RBs=front_RBs, inf=inf, nf=nf)
        self.EventSubHead_2 = EventSubHead(front_RBs=front_RBs, inf=inf, nf=nf)

    def forward(self, event, fea1, fea2):
        num_inter = event.shape[1]
        image_feature = []
        image_feature.append(fea1[0])
        for idx in range(num_inter):
            # inverse events
            lr_event_t = event[:, idx, ...]
            event_inverse_t = lr_event_t[:, [1,0], ...]
            event_inverse_t = torch.neg(event_inverse_t)

            # extract inversed events features
            lr_event_t = self.EventSubHead_1(lr_event_t)
            event_inverse_t = self.EventSubHead_2(event_inverse_t)
            
            image_feature.append(self.epcd_align(
                fea1, fea2, lr_event_t, event_inverse_t
            ))
        image_feature.append(fea2[0])
        return torch.stack(image_feature, dim=1)
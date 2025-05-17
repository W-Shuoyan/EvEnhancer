import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY


from .arch_util import make_layer, ResidualBlockNoBN
from .ema_modules import EMA_Encoder
from .recurrent_sub_modules import BRCLayer
from .decoder import LIVT_Decoder


@ARCH_REGISTRY.register()
class EvEnhancerArch(nn.Module):
    def __init__(
            self,
            event_channels,
            channels,
            n_feats,
            front_RBs,
            base_dim,
            head,
            r,
            r_t
            ):
        super(EvEnhancerArch, self).__init__()

        # User Setting
        self.n_feats = n_feats

        # Network Part
        self.image_head = nn.Conv2d(
            channels, n_feats, 5, padding=2)
        self.event_head = nn.Conv2d(
            event_channels, n_feats, 5, padding=2)
        
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, front_RBs, num_feat=n_feats)
        self.fea_L2_conv1 = nn.Conv2d(n_feats, n_feats, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(n_feats, n_feats, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # EMA
        self.ema = EMA_Encoder(front_RBs=front_RBs, inf=event_channels, nf=n_feats)

        # BRC
        self.backward_brclayer = BRCLayer(n_feats, n_feats, fuse_two_direction=False, use_atten_fuse=True)
        self.forward_brclayer = BRCLayer(n_feats, n_feats, fuse_two_direction=True, use_atten_fuse=True)

        # LIVT
        self.decoder = LIVT_Decoder(in_dim=n_feats,
                              base_dim=base_dim,
                              head=head,
                              r=r,
                              r_t=r_t
                              )

    def forward(self, image, event, scale, times):
        if len(image.shape) == 4:
            image.unsqueeze(0)
        if len(event.shape) == 4:
            event.unsqueeze(0)
        image_b, image_t, image_c, image_h, image_w = image.shape
        event_b, event_t, event_c, event_h, event_w = event.shape

        # Head
        image_head_feature = self.image_head(
            image.view(-1, image_c, image_h, image_w))
        event_head_feature = self.event_head(
            event.view(-1, event_c, event_h, event_w))
        event_head_feature = event_head_feature.view(event_b, event_t, -1, event_h, event_w)
        
        # extract LR frames features
        # L1
        L1_fea = self.lrelu(image_head_feature)
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(image_b, image_t, -1, image_h, image_w)
        L2_fea = L2_fea.view(image_b, image_t, -1, L2_fea.shape[2], L2_fea.shape[3])
        L3_fea = L3_fea.view(image_b, image_t, -1, L3_fea.shape[2], L3_fea.shape[3])
  
        fea1 = [
            L1_fea[:, 0, :, :, :].clone(), L2_fea[:, 0, :, :,
                                                  :].clone(), L3_fea[:, 0, :, :, :].clone()
        ]
        fea2 = [
            L1_fea[:, 1, :, :, :].clone(), L2_fea[:, 1, :, :,
                                                  :].clone(), L3_fea[:, 1, :, :, :].clone()
        ]
        del L1_fea, L2_fea, L3_fea
        torch.cuda.empty_cache()

        # EMA
        image_feature = self.ema(event, fea1, fea2)
        T = image_feature.shape[1]

        # backward propagation
        backward_states = []
        for frame_idx in range(T-1, -1, -1):
            image_cur = image_feature[:, frame_idx, :, :, :]
            if frame_idx == T-1:
                event_cur = None
                _, state = self.backward_brclayer(x=image_cur,y=event_cur, prev_state=None)
            else:
                if frame_idx == 0:
                    event_cur = None
                else:
                    event_cur = event_head_feature[:, frame_idx-1, :, :, :]
                _, state = self.backward_brclayer(x=image_cur,y=event_cur, prev_state=state)
            backward_states.append(state)

        # forward propagation
        pro_feature = []
        for frame_idx in range(0, T):
            image_cur = image_feature[:, frame_idx, :, :, :]
            if frame_idx == 0:
                event_cur = None
                x, state = self.forward_brclayer(x=image_cur,y=event_cur, prev_state=None, bi_direction_state=backward_states[T-1-frame_idx])
            else:
                if frame_idx == T-1:
                    event_cur = None
                else:
                    event_cur = event_head_feature[:, frame_idx-1, :, :, :]
                x, state = self.forward_brclayer(x=image_cur,y=event_cur, prev_state=state, bi_direction_state=backward_states[T-1-frame_idx])
            pro_feature.append(x)

        pro_feature = torch.stack(pro_feature, dim=1)

        out = pro_feature + image_feature
        out = out.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
        del pro_feature, image_feature
        torch.cuda.empty_cache()

        # LIVT
        out = self.decoder(out, scale, times)
        out = torch.stack(out, dim=1)
        return out  # b,times,c,h,w

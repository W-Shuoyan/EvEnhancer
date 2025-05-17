import torch
import torch.nn as nn

from .module_util import ConvLayer, ChannelAtention, SimpleRecurrentConv


class BRCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False, use_atten_fuse=False):
        super(BRCLayer, self).__init__()
        self.relu_slope = relu_slope
        self.use_atten_fuse = use_atten_fuse

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        if self.use_atten_fuse:
            self.atten_fuse = ChannelAtention(c=in_channels, c_out = out_channels, DW_Expand=1, FFN_Expand=2)

        self.recurrent_block = SimpleRecurrentConv(out_channels, out_channels, num_block=num_block)
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope, norm)

    def forward(self, x, y=None, prev_state=None, bi_direction_state = None):
        if y is not None:
            if self.use_atten_fuse:
                x = self.atten_fuse(x, y)
            else:
                x = x + y
                x = self.conv(x)
                if self.relu_slope is not None:
                    x = self.relu(x)
        else:
            x = self.conv(x)
            if self.relu_slope is not None:
                x = self.relu(x)

        x, state = self.recurrent_block(x, prev_state)
        if bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)
        
        return x, state
    


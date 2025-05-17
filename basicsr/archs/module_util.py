import torch
import torch.nn as nn
import torch.nn.init as init
from .arch_util import make_layer, ResidualBlockNoBN


class PositionEncoder3d(nn.Module):
    def __init__(
        self,
        posenc_scale=10,
        enc_dims=64,
        head=1,
        gamma=1
    ):
        super().__init__()

        self.posenc_scale = posenc_scale

        self.enc_dims = enc_dims
        self.head = head
        self.gamma = gamma

        self.define_parameter()

    def define_parameter(self):
        self.b_vals = 2.**torch.linspace(
            0, self.posenc_scale, self.enc_dims // 4
        ) - 1  # -1 -> (2 * pi)
        self.b_vals = torch.stack([self.b_vals, torch.zeros_like(self.b_vals), torch.zeros_like(self.b_vals)], dim=-1)
        self.b_vals = torch.cat([self.b_vals, torch.roll(self.b_vals, 1, -1)], dim=0)
        self.a_vals = torch.ones(self.b_vals.shape[0])
        self.proj = nn.Linear(self.enc_dims, self.head)


    def init_weight(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, positions):
        self.b_vals = self.b_vals.cuda()
        self.a_vals = self.a_vals.cuda()

        sin_part = self.a_vals * torch.sin(
            torch.matmul(positions, self.b_vals.transpose(-2, -1))
        )
        cos_part = self.a_vals * torch.cos(
            torch.matmul(positions, self.b_vals.transpose(-2, -1))
        )
        pos_enocoding = torch.cat([sin_part, cos_part], dim=-1)
        pos_bias = self.proj(pos_enocoding)

        return pos_enocoding, pos_bias


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    

class ChannelAtention(nn.Module):
    def __init__(self, c, c_out, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv1_e = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2_e = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=2*dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        # Channel Attention
        self.se_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c_out, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_y_side = nn.Conv2d(in_channels=c, out_channels=c_out, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm1_e = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c_out, 1, 1)), requires_grad=True)

    def forward(self, potential_feat, event_feat):
        x = potential_feat
        x_e = event_feat
        x = self.norm1(x)
        x_e = self.norm1_e(x_e)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x_e = self.conv1_e(x_e)
        x_e = self.conv2_e(x_e)
        x_e = self.gelu(x_e)
        x = x * self.se_1(x)
        x_e = x * self.se_2(x_e)
        x = torch.cat((x_e, x), dim=1) # cat in c
        x = self.conv3(x) # fuse
        x = self.dropout1(x)
        y = potential_feat + event_feat + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        y = self.conv_y_side(y)
        return y + x_e * self.gamma

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, relu_slope=0.2, norm=None):
        super(ConvLayer, self).__init__()
        self.relu_slope = relu_slope

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if relu_slope is not None:
            if type(relu_slope) is str:
                self.relu = nn.ReLU()
            else:
                self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.relu_slope is not None:
            out = self.relu(out)

        return out


class SimpleRecurrentConv(nn.Module):
    def __init__(self, input_size, hidden_size, num_block=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.forward_trunk = ConvResidualBlocks(input_size + hidden_size, input_size, num_block)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, prev_state):

        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(x.device)

        # backward branch
        feat_prop = torch.cat([x, prev_state], dim=1)
        feat_prop = self.forward_trunk(feat_prop)
        state = feat_prop

        return feat_prop, state
    

class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)
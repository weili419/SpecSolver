import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .Transolver_utils.utils import timestep_embedding
from .Transolver_utils.utils import Physics_Attention_Structured_Mesh_2D

import math
from argparse import Namespace
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import os
import cv2
from skimage.restoration import unwrap_phase
ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


# def make_coord(shape, ranges=None, flatten=True):
#     """ Make coordinates at grid centers.
#     """
#     coord_seqs = []
#     for i, n in enumerate(shape):
#         if ranges is None:
#             v0, v1 = -1, 1
#         else:
#             v0, v1 = ranges[i]
#         r = (v1 - v0) / (2 * n)
#         seq = v0 + r + (2 * r) * torch.arange(n).float()
#         coord_seqs.append(seq)
#     ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
#     if flatten:
#         ret = ret.view(-1, ret.shape[-1])
#     return ret

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# modified from: https://github.com/thstkdgus35/EDSR-PyTorch
url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}    

class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        #x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1, n_colors=1,
                       scale=2, no_upsampling=True, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = n_colors
    return EDSR(args)

#################################################################################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes//4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpaPyBlock(nn.Module):
    def __init__(self, inchannels, outchannels, bias=True):
        super(SpaPyBlock, self).__init__()
        #dim = inchannels//2
        self.scale1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        )
        self.scale1_2 = nn.Sequential(
            nn.Upsample(mode='bilinear',scale_factor=1/2),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        )
        self.scale1_4 = nn.Sequential(
            nn.Upsample(mode='bilinear',scale_factor=1/4),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        )
        self.channel = ChannelAttention(inchannels*3)
        self.out = nn.Sequential(
            nn.Conv2d(inchannels*3, outchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias),
            nn.LeakyReLU(0.2)
            )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        y1 = self.scale1(x)
        y2 = self.up2(self.scale1_2(x))
        y3 = self.up4(self.scale1_4(x))
        y = torch.cat((y1,y2,y3),dim=1)
        y = self.channel(y)*y
        y = self.out(y)   
        return y + x
    
class SpePyBlock(nn.Module):
    def __init__(self, inchannels, bias=True):
        super(SpePyBlock, self).__init__()
        self.conv2 = nn.Sequential(
                        nn.Conv2d(inchannels, inchannels*2, kernel_size=3, stride=1, padding=1, groups=2, bias=bias),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(inchannels*2, inchannels*2, kernel_size=3, stride=1, padding=1, groups=2, bias=bias),
                        nn.LeakyReLU(0.2)
                        )

        self.conv4 = nn.Sequential(
                        nn.Conv2d(inchannels, inchannels*2, kernel_size=3, stride=1, padding=1, groups=4, bias=bias),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(inchannels*2, inchannels*2, kernel_size=3, stride=1, padding=1, groups=4, bias=bias),
                        nn.LeakyReLU(0.2)
                        )
        
        self.conv8 = nn.Sequential(
                        nn.Conv2d(inchannels, inchannels*2, kernel_size=3, stride=1, padding=1, groups=8, bias=bias),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(inchannels*2, inchannels*2, kernel_size=3, stride=1, padding=1, groups=8, bias=bias),
                        nn.LeakyReLU(0.2)
                        )

    def forward(self, x2,x4,x8):
        _, c, _, _ = x2.shape
        if c % 8 != 0:
            x2 = torch.cat((x2,x2[:,c-(8-c % 8)-1:c-1,:,:]), dim=1)
            x4 = torch.cat((x4,x4[:,c-(8-c % 8)-1:c-1,:,:]), dim=1)
            x8 = torch.cat((x8,x8[:,c-(8-c % 8)-1:c-1,:,:]), dim=1)
        x2_1 = self.conv2(x2)
        x4_1 = self.conv4(x4)
        x8_1 = self.conv8(x8)

        return x2_1,x4_1,x8_1

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

        )

    def forward(self, x):
        return self.double_conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_qm = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_km = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_vm = nn.Linear(dim, dim_head * heads, bias=False)        
        self.to_k2 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v2 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k4 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v4 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k8 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v8 = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescalem = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale2 = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale4 = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale8 = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_2,x_4,x_8,y):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = y.shape
        x2 = x_2.reshape(b,h*w,c)
        x4 = x_4.reshape(b,h*w,c)
        x8 = x_8.reshape(b,h*w,c)
        y = y.reshape(b,h*w,c)

        q_inpm = self.to_qm(y)
        k_inpm = self.to_km(y)
        v_inpm = self.to_vm(y)
        qm, km, vm = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inpm, k_inpm, v_inpm))
        vm = vm
        # q: b,heads,hw,c
        qm = qm.transpose(-2, -1)
        km = km.transpose(-2, -1)
        vm = vm.transpose(-2, -1)
        qm = F.normalize(qm, dim=-1, p=2)
        km = F.normalize(km, dim=-1, p=2)
        attnm = (km @ qm.transpose(-2, -1))   # A = K^T*Q
        attnm = attnm * self.rescalem
        attnm = attnm.softmax(dim=-1)

        k_inp2 = self.to_k2(x2)
        v_inp2 = self.to_v2(x2)
        k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (k_inp2, v_inp2))
        v2 = v2
        # q: b,heads,hw,c
        k2 = k2.transpose(-2, -1)
        v2 = v2.transpose(-2, -1)
        k2 = F.normalize(k2, dim=-1, p=2)
        attn2 = (k2 @ qm.transpose(-2, -1))   # A = K^T*Q
        attn2 = attn2 * self.rescale2
        attn2 = attn2.softmax(dim=-1)

        k_inp4 = self.to_k4(x4)
        v_inp4 = self.to_v4(x4)
        k4, v4 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (k_inp4, v_inp4))
        v4 = v4
        # q: b,heads,hw,c
        k4 = k4.transpose(-2, -1)
        v4 = v4.transpose(-2, -1)
        k4 = F.normalize(k4, dim=-1, p=2)
        attn4 = (k4 @ qm.transpose(-2, -1))   # A = K^T*Q
        attn4 = attn4 * self.rescale4
        attn4 = attn4.softmax(dim=-1)

        k_inp8 = self.to_k8(x8)
        v_inp8 = self.to_v8(x8)
        k8, v8 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (k_inp8, v_inp8))
        v8 = v8
        # q: b,heads,hw,c
        k8 = k8.transpose(-2, -1)
        v8 = v8.transpose(-2, -1)
        k8 = F.normalize(k8, dim=-1, p=2)
        attn8 = (k8 @ qm.transpose(-2, -1))   # A = K^T*Q
        attn8 = attn8 * self.rescale4
        attn8 = attn8.softmax(dim=-1)

        x = attnm @ vm +  attn2 @ v2 + attn4 @ v4 + attn8 @ v8  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inpm.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)
    
class MLSIF(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x_2,x_4,x_8,y):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x_2 = x_2.permute(0, 2, 3, 1)
        x_4 = x_4.permute(0, 2, 3, 1)
        x_8 = x_8.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x_2,x_4,x_8,y) + y
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out
       
#################################################################################################################################    
class MLP_3(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Sequential(
                        nn.Conv2d(lastv, hidden, kernel_size=1, bias=False),
                        nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False, groups=hidden),
                        ))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Sequential(
                        nn.Conv2d(lastv, out_dim, kernel_size=1, bias=False),
                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False, groups=out_dim),
                        ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class MLP_1(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

################################################################################################################################# 
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x




class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            H=85,
            W=85
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num, H=H, W=W)
        # self.GI = Galerkin_integral(128, 8)
        # self.PR = Progressive_Resampling_integral(dim=128, input_resolution=1024, num_heads=8, ssl=16)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        
        # x_mid_slice = fx[0, :, :]
        # x_mid_slice = x_mid_slice.view(512, 512, 64)
        # cmap = plt.cm.viridis  # 使用 viridis 作为配色，类似自然色调的配色
        # for i in range(32):
        #     # 选择第 i 个通道
        #     channel_image = x_mid_slice[:, :, i].cpu().detach().numpy()
        #     channel_image = (channel_image - np.min(channel_image)) / (np.max(channel_image) - np.min(channel_image))
        #     gamma = 4.0  # 伽马值，值越大，亮部分越亮，暗部分越暗
        #     enhanced_image = np.power(channel_image, gamma)

        #     # 显示和保存图片
        #     plt.imshow(enhanced_image, cmap=cmap)
        #     # plt.title(f"Channel {i+1}")
        #     plt.axis('off')
        #     plt.savefig(f"/root/data1/CAVE/Figure/channel_{i+1}.png", bbox_inches='tight', pad_inches=0)
        #     plt.show()

        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.cuda()) - torch.fft.fft2(y.cuda())
        loss = torch.mean(abs(diff))
        return loss

class Modelcave(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 H=85,
                 W=85,
                 ):
        super(Modelcave, self).__init__()
        self.__name__ = 'Transolver_2D'
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        self.out_dim = out_dim

        spa_edsr_num = 6 #5#6
        guide_dim = 128 #96#128
        hsi_dim = 3
        msi_dim = 31
        self.spatial_encoder = make_edsr_baseline(n_resblocks=spa_edsr_num, n_feats=guide_dim, n_colors=hsi_dim+msi_dim)
        ##########################################################################################################################
        self.spa1 = SpaPyBlock(guide_dim, 128)
        # # self.spe1 = SpePyBlock(32)
        # # self.inc = DoubleConv(msi_dim+hsi_dim, 64)
        # # self.mls1  = MLSIF(dim=64, num_blocks=1, dim_head=64, heads=64 // 64)
        # ##########################################################################################################################
        imnet_in_dim = 128
        NIR_dim = 128
        mlp_dim = [128]
        self.imnet1 = MLP_1(imnet_in_dim, out_dim=NIR_dim, hidden_list=mlp_dim)

        # self.imnet_3_3 = MLP_3(imnet_in_dim, out_dim=NIR_dim, hidden_list=mlp_dim)
        self.imnet2 = MLP_3(imnet_in_dim, out_dim=NIR_dim, hidden_list=mlp_dim)
        ##########################################################################################################################
        ##########################################################################################################################
        if self.unified_pos:
            # self.pos = self.get_grid()
            self.preprocess = MLP(fun_dim + 2, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        # else:
        #     self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        # self.Time_Input = Time_Input
        # self.n_hidden = n_hidden
        # self.space_dim = space_dim
        # if Time_Input:
        #     self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      H=H,
                                                      W=W,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        # self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, up_LR, RGB, HR=None, j=None, LR =None, T=None):

        fx = torch.cat((up_LR, RGB), 1) # 连接 [B, C, H, W]
        fx = self.spatial_encoder(fx) # EDSR-encoder
        B, C, H, W = fx.shape
        
        fxHR = torch.cat((HR, RGB), 1)
        fxHR_spa = self.spatial_encoder(fxHR)
        #########################################################
        if j is not None:
            yuan = fxHR_spa[0, 25, :, :]  # shape: [512, 512]
            # 对图像做傅里叶变换
            img_ffted2 = torch.fft.fft2(yuan)
            fft_shifted2 = torch.fft.fftshift(img_ffted2)

            # 获取幅度和相位
            img_mag2 = torch.abs(fft_shifted2)
            amplitude_log2 = torch.log(1 + img_mag2)
            # img_pha2 = torch.angle(fft_shifted2)

            gamma = 1.5
            amplitude_log2 = torch.pow(amplitude_log2, gamma)
            amplitude_normalized2 = (amplitude_log2 - torch.min(amplitude_log2)) / (torch.max(amplitude_log2) - torch.min(amplitude_log2))
            
            # phase_normalized = (img_pha + torch.pi) / (2 * torch.pi)  # 相位从 [-π, π] 映射到 [0, 1]
            # phase_mean = torch.mean(phase_normalized)      
            # _, phase_binary = cv2.threshold(phase_normalized.cpu().numpy(), phase_mean.item(), 1.0, cv2.THRESH_BINARY)
############################################################################################################################            
            y1 = self.spa1(fx)

            feat_ffted = torch.fft.fftn(fx, dim=(-2,-1))
            feat_mag = torch.abs(feat_ffted).permute(0,2,3,1).reshape(B*H*W,-1)
            # feat_mag = torch.abs(feat_ffted)
            feat_pha = torch.angle(feat_ffted)

            ffted_mag = self.imnet1(feat_mag).reshape(B, C, H, W)
            # ffted_mag = self.imnet_3_3(feat_mag)
            ffted_pha = self.imnet2(feat_pha)

            real = ffted_mag * torch.cos(ffted_pha)
            imag = ffted_mag * torch.sin(ffted_pha)        
            ffted = torch.complex(real, imag)
            output = torch.fft.ifftn(ffted, dim =(-2,-1))
            output = torch.abs(output) + y1
############################################################################################################################
            img = output[0, 25, :, :]  # shape: [512, 512]
            img_ffted = torch.fft.fft2(img)
            fft_shifted = torch.fft.fftshift(img_ffted)

            # 获取幅度和相位
            img_mag = torch.abs(fft_shifted)
            amplitude_log = torch.log(1 + img_mag)
            # img_pha = torch.angle(fft_shifted)

            gamma = 1.5
            amplitude_log = torch.pow(amplitude_log, gamma)
            amplitude_normalized = (amplitude_log - torch.min(amplitude_log)) / (torch.max(amplitude_log) - torch.min(amplitude_log))
            
            # phase_normalized = (img_pha + torch.pi) / (2 * torch.pi)  # 相位从 [-π, π] 映射到 [0, 1]
            # phase_mean = torch.mean(phase_normalized)      
            # _, phase_binary = cv2.threshold(phase_normalized.cpu().numpy(), phase_mean.item(), 1.0, cv2.THRESH_BINARY)   
############################################################################################################################ 
            difference = amplitude_normalized2 - amplitude_normalized
            diff_np = difference.cpu().detach().numpy() if difference.is_cuda else difference.detach().numpy()

            save_path = "/root/data1/CAVE/experiment/Debug/4/1_1conv/3_3.png"  # 保存路径
            plt.imshow(amplitude_normalized.cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

            save_path2 = "/root/data1/CAVE/experiment/Debug/4/1_1conv/3_3error.png"  # 保存路径
            plt.imshow(diff_np, cmap='bwr', vmin=-1, vmax=1)
            plt.axis('off')
            plt.savefig(save_path2, bbox_inches='tight')
            plt.close()


            # for i in range(fx.shape[1]):  # 遍历通道
            #     i = 25
            #     save_path = "/root/data1/CAVE/experiment/Debug/4/testimage"
            #     img = fx21[0, i, :, :]  # shape: [512, 512]

            #     # 对图像做傅里叶变换
            #     img_ffted = torch.fft.fft2(img)
            #     fft_shifted = torch.fft.fftshift(img_ffted)

            #     # 获取幅度和相位
            #     ffted_mag = torch.abs(fft_shifted)
            #     amplitude_log = torch.log(1 + ffted_mag)
            #     ffted_pha = torch.angle(fft_shifted)

            #     ffted_mag2 = torch.abs(img_ffted)
            #     ffted_pha2 = torch.angle(img_ffted)               
            #     # 去除相位
            #     # 去除相位（保留振幅，设置相位为0）
            #     # real_amp_removed = ffted_mag2 * torch.cos(torch.zeros_like(ffted_pha2))  # 相位为0
            #     # imag_amp_removed = ffted_mag2 * torch.sin(torch.zeros_like(ffted_pha2))
            #     real_amp_removed = ffted_mag2  # 只保留振幅信息，实部就是振幅
            #     imag_amp_removed = torch.zeros_like(ffted_mag2)  # 虚部设为0

            #     # 重构复数傅里叶变换（去除相位后的复数表示）
            #     ffted_amp_removed = torch.complex(real_amp_removed, imag_amp_removed)
            #     ffted_amp_removed = torch.fft.ifftn(ffted_amp_removed).real
            #     # ffted_amp_removed = torch.abs(ffted_amp_removed)
                
            #     ffted_amp_removed = (ffted_amp_removed - torch.min(ffted_amp_removed)) / (torch.max(ffted_amp_removed) - torch.min(ffted_amp_removed))

            #     # 去除振幅（保留相位，振幅设置为0）
            #     real_pha_removed = torch.cos(ffted_pha2)  # 振幅为0，保留相位
            #     imag_pha_removed = torch.sin(ffted_pha2)  # 振幅为0，保留相位

            #     # 重构复数傅里叶变换（去除振幅后的复数表示）
            #     ffted_pha_removed = torch.complex(real_pha_removed, imag_pha_removed)
            #     ffted_pha_removed = torch.fft.ifftn(ffted_pha_removed).real
            #     # ffted_pha_removed = torch.abs(ffted_pha_removed)

            #     ffted_pha_removed = (ffted_pha_removed - torch.min(ffted_pha_removed)) / (torch.max(ffted_pha_removed) - torch.min(ffted_pha_removed))

            #     gamma = 1.5
            #     amplitude_log = torch.pow(amplitude_log, gamma)
        
            #     amplitude_normalized = (amplitude_log - torch.min(amplitude_log)) / (torch.max(amplitude_log) - torch.min(amplitude_log))
            #     phase_normalized = (ffted_pha + torch.pi) / (2 * torch.pi)  # 相位从 [-π, π] 映射到 [0, 1]

            #     phase_mean = torch.mean(phase_normalized)      
            #     _, phase_binary = cv2.threshold(phase_normalized.cpu().numpy(), phase_mean.item(), 1.0, cv2.THRESH_BINARY)
        
            #     fig, axes = plt.subplots(3, 2, figsize=(12, 18))  # 3行2列
            #     axes[0, 0].imshow(img.cpu().numpy(), cmap='gray')
            #     axes[0, 0].set_title('Original Image')
            #     axes[0, 0].axis('off')
            #     # 振幅图
            #     axes[1, 0].imshow(amplitude_normalized.cpu().numpy(), cmap='gray')
            #     axes[1, 0].set_title('Amplitude Spectrum')
            #     axes[1, 0].axis('off')

            #     # 相位图
            #     axes[1, 1].imshow(phase_normalized.cpu().numpy(), cmap='gray')
            #     axes[1, 1].set_title('Phase Spectrum')
            #     axes[1, 1].axis('off')

            #     # 振幅后逆傅里叶变换得到的图像
            #     axes[2, 0].imshow(ffted_amp_removed.cpu().numpy(), cmap='gray')
            #     axes[2, 0].set_title('IFFT amp')
            #     axes[2, 0].axis('off')

            #     # 相位后逆傅里叶变换得到的图像
            #     axes[2, 1].imshow(ffted_pha_removed.cpu().numpy(), cmap='gray')
            #     axes[2, 1].set_title('IFFT pha')
            #     axes[2, 1].axis('off')

            #     combined_path = os.path.join(save_path, f"combined_channel_{i}_{j}.png")  
            #     plt.savefig(combined_path, bbox_inches='tight')
            #     plt.close()
            #     return up_LR

            for i in range(fx.shape[1]):  # 遍历通道
                save_path = "/root/data1/CAVE/experiment/Debug/4/image"
                img = fx[0, 25, :, :]  # shape: [512, 512]

                # 对图像做傅里叶变换
                img_ffted = torch.fft.fft2(img)
                fft_shifted = torch.fft.fftshift(img_ffted)

                # 获取幅度和相位
                img_mag = torch.abs(fft_shifted)
                amplitude_log = torch.log(1 + img_mag)
                img_pha = torch.angle(fft_shifted)

                gamma = 1.5
                amplitude_log = torch.pow(amplitude_log, gamma)
        
                amplitude_normalized = (amplitude_log - torch.min(amplitude_log)) / (torch.max(amplitude_log) - torch.min(amplitude_log))
                phase_normalized = (img_pha + torch.pi) / (2 * torch.pi)  # 相位从 [-π, π] 映射到 [0, 1]

                phase_mean = torch.mean(phase_normalized)      
                _, phase_binary = cv2.threshold(phase_normalized.cpu().numpy(), phase_mean.item(), 1.0, cv2.THRESH_BINARY)

                # 保存振幅图
                fig_amp = plt.figure(figsize=(6, 6))  # 单独设置尺寸
                plt.imshow(amplitude_normalized.cpu().numpy(), cmap='gray')
                plt.axis('off')
                amplitude_path = os.path.join(save_path, f"amplitude_channel_{25}_{j}.png")
                plt.savefig(amplitude_path, bbox_inches='tight')
                plt.close(fig_amp)  # 显式关闭当前figure

                # 保存相位图
                fig_phase = plt.figure(figsize=(6, 6))  # 单独设置尺寸
                plt.imshow(phase_normalized.cpu().numpy(), cmap='gray')
                plt.axis('off')
                phase_path = os.path.join(save_path, f"phase_channel_{25}_{j}.svg")
                plt.savefig(phase_path, bbox_inches='tight')
                plt.close(fig_phase)  # 显式关闭当前figure                

                center_h, center_w = 256, 256  # 得到中心坐标
                half_size = 64  # 128的一半
                center_region = phase_normalized[center_h - half_size:center_h + half_size,
                                            center_w - half_size:center_w + half_size].cpu().numpy()

                # 放大四倍，即输出尺寸为128*4 x 128*4 = 512x512
                center_zoomed = cv2.resize(center_region, (128 * 4, 128 * 4), interpolation=cv2.INTER_NEAREST)

                # 保存放大后的相位图
                fig_zoom = plt.figure(figsize=(6, 6))
                plt.imshow(center_zoomed, cmap='gray')
                plt.axis('off')
                zoom_path = os.path.join(save_path, f"phase_channel_zoom_{25}_{j}.svg")
                plt.savefig(zoom_path, bbox_inches='tight')
                plt.close(fig_zoom)

            # for i in range(fx2.shape[1]):  # 遍历通道
            #     save_path = "/root/data1/CAVE/experiment/Debug/4/HR_RGB_image"
            #     img = fx2[0, 25, :, :]  # shape: [512, 512]

            #     # 对图像做傅里叶变换
            #     img_ffted = torch.fft.fft2(img)
            #     fft_shifted = torch.fft.fftshift(img_ffted)

            #     # 获取幅度和相位
            #     img_mag = torch.abs(fft_shifted)
            #     amplitude_log = torch.log(1 + img_mag)
            #     img_pha = torch.angle(fft_shifted)

            #     gamma = 1.5
            #     amplitude_log = torch.pow(amplitude_log, gamma)
        
            #     amplitude_normalized = (amplitude_log - torch.min(amplitude_log)) / (torch.max(amplitude_log) - torch.min(amplitude_log))
            #     phase_normalized = (img_pha + torch.pi) / (2 * torch.pi)  # 相位从 [-π, π] 映射到 [0, 1]

            #     phase_mean = torch.mean(phase_normalized)      
            #     _, phase_binary = cv2.threshold(phase_normalized.cpu().numpy(), phase_mean.item(), 1.0, cv2.THRESH_BINARY)
        
            #     # 保存振幅图
            #     fig_amp = plt.figure(figsize=(6, 6))  # 单独设置尺寸
            #     plt.imshow(amplitude_normalized.cpu().numpy(), cmap='gray')
            #     plt.axis('off')
            #     amplitude_path = os.path.join(save_path, f"amplitude_channel_{25}_{j}.png")
            #     plt.savefig(amplitude_path, bbox_inches='tight')
            #     plt.close(fig_amp)  # 显式关闭当前figure

            #     # 保存相位图
            #     fig_phase = plt.figure(figsize=(6, 6))  # 单独设置尺寸
            #     plt.imshow(phase_normalized.cpu().numpy(), cmap='gray')
            #     plt.axis('off')
            #     phase_path = os.path.join(save_path, f"phase_channel_{25}_{j}.svg")
            #     plt.savefig(phase_path, bbox_inches='tight')
            #     plt.close(fig_phase)  # 显式关闭当前figure     

            #     center_h, center_w = 256, 256  # 得到中心坐标
            #     half_size = 64  # 128的一半
            #     center_region = phase_normalized[center_h - half_size:center_h + half_size,
            #                                 center_w - half_size:center_w + half_size].cpu().numpy()

            #     # 放大四倍，即输出尺寸为128*4 x 128*4 = 512x512
            #     center_zoomed = cv2.resize(center_region, (128 * 4, 128 * 4), interpolation=cv2.INTER_NEAREST)

            #     # 保存放大后的相位图
            #     fig_zoom = plt.figure(figsize=(6, 6))
            #     plt.imshow(center_zoomed, cmap='gray')
            #     plt.axis('off')
            #     zoom_path = os.path.join(save_path, f"phase_channel_zoom_{25}_{j}.svg")
            #     plt.savefig(zoom_path, bbox_inches='tight')
            #     plt.close(fig_zoom)

            return up_LR
        
        # fxHR_spa_ffted = torch.fft.fftn(fxHR_spa, dim=(-2,-1))
        # fxHR_spa_mag = torch.abs(fxHR_spa_ffted)
        #########################################################
        # 我需要再加一个空间域
        # 用DSPNet的试一下
        y1 = self.spa1(fx)
        # z1_2, z1_4, z1_8 = self.spe1(up_LR, up_LR, up_LR)
        # feat = self.inc(torch.cat((up_LR, RGB), dim=1))
        # x1 = self.mls1(z1_2, z1_4, z1_8, feat)
        # fx = torch.cat((y1, x1), 1)
        #########################################################
        # 傅里叶域内
        feat_ffted = torch.fft.fftn(fx, dim=(-2,-1))
        # # 获取幅度和相位
        feat_mag = torch.abs(feat_ffted).permute(0,2,3,1).reshape(B*H*W,-1)
        # feat_mag = torch.abs(feat_ffted)############################################
        feat_pha = torch.angle(feat_ffted)

        ffted_mag = self.imnet1(feat_mag).reshape(B, C, H, W)
        # ffted_mag = self.imnet_3_3(feat_mag)
        ffted_pha = self.imnet2(feat_pha)

        real = ffted_mag * torch.cos(ffted_pha)
        imag = ffted_mag * torch.sin(ffted_pha)        
        ffted = torch.complex(real, imag)
        output = torch.fft.ifftn(ffted, dim =(-2,-1))
        output = torch.abs(output)
        middleoutput = y1 + output
        fx = middleoutput
        #########################################################
        fx = fx.permute(0,2,3,1)
        # B, H, W, C = fx.shape
        fx = fx.reshape(B, -1, C)

        if self.unified_pos:
            # x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref) # 不用源代码的位置信息嵌入
            x = x.reshape(B, H*W, -1) # 用原来自带的位置编码效果就还不错的，比上面那个好
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        # else:
        #     fx = self.preprocess(x)
        #     fx = fx + self.placeholder[None, None, :]

        # if T is not None:
        #     Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
        #     Time_emb = self.time_fc(Time_emb)
        #     fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        
        fx = fx.reshape(B, H, W, self.out_dim).permute(0,3,1,2)

        return fx + up_LR, ffted_mag, ffted_mag

class Modelharvard(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 H=85,
                 W=85,
                 ):
        super(Modelharvard, self).__init__()
        self.__name__ = 'Transolver_2D'
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        self.out_dim = out_dim

        spa_edsr_num = 5
        guide_dim = 96
        hsi_dim = 3
        msi_dim = 31
        self.spatial_encoder = make_edsr_baseline(n_resblocks=spa_edsr_num, n_feats=guide_dim, n_colors=hsi_dim+msi_dim)
        ##########################################################################################################################
        self.spa1 = SpaPyBlock(guide_dim, 96)
        # # self.spe1 = SpePyBlock(32)
        # # self.inc = DoubleConv(msi_dim+hsi_dim, 64)
        # # self.mls1  = MLSIF(dim=64, num_blocks=1, dim_head=64, heads=64 // 64)
        # ##########################################################################################################################
        imnet_in_dim = 96
        NIR_dim = 96
        mlp_dim = []
        self.imnet1 = MLP_1(imnet_in_dim, out_dim=NIR_dim, hidden_list=mlp_dim)

        # self.imnet_3_3 = MLP_3(imnet_in_dim, out_dim=NIR_dim, hidden_list=mlp_dim)
        self.imnet2 = MLP_3(imnet_in_dim, out_dim=NIR_dim, hidden_list=mlp_dim)
        ##########################################################################################################################
        ##########################################################################################################################
        if self.unified_pos:
            # self.pos = self.get_grid()
            self.preprocess = MLP(fun_dim + 2, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        # else:
        #     self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        # self.Time_Input = Time_Input
        # self.n_hidden = n_hidden
        # self.space_dim = space_dim
        # if Time_Input:
        #     self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      H=H,
                                                      W=W,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        # self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, up_LR, RGB, HR=None, j=None, LR =None, T=None):

        fx = torch.cat((up_LR, RGB), 1) # 连接 [B, C, H, W]
        fx = self.spatial_encoder(fx) # EDSR-encoder
        B, C, H, W = fx.shape
        
        # fxHR = torch.cat((HR, RGB), 1)
        # fxHR_spa = self.spatial_encoder(fxHR)
        #########################################################
     
        # fxHR_spa_ffted = torch.fft.fftn(fxHR_spa, dim=(-2,-1))
        # fxHR_spa_mag = torch.abs(fxHR_spa_ffted)
        #########################################################
        # 我需要再加一个空间域
        # 用DSPNet的试一下
        y1 = self.spa1(fx)
        # z1_2, z1_4, z1_8 = self.spe1(up_LR, up_LR, up_LR)
        # feat = self.inc(torch.cat((up_LR, RGB), dim=1))
        # x1 = self.mls1(z1_2, z1_4, z1_8, feat)
        # fx = torch.cat((y1, x1), 1)
        #########################################################
        # 傅里叶域内
        feat_ffted = torch.fft.fftn(fx, dim=(-2,-1))
        # # 获取幅度和相位
        feat_mag = torch.abs(feat_ffted).permute(0,2,3,1).reshape(B*H*W,-1)
        # feat_mag = torch.abs(feat_ffted)############################################
        feat_pha = torch.angle(feat_ffted)

        ffted_mag = self.imnet1(feat_mag).reshape(B, C, H, W)
        # ffted_mag = self.imnet_3_3(feat_mag)
        ffted_pha = self.imnet2(feat_pha)

        real = ffted_mag * torch.cos(ffted_pha)
        imag = ffted_mag * torch.sin(ffted_pha)        
        ffted = torch.complex(real, imag)
        output = torch.fft.ifftn(ffted, dim =(-2,-1))
        output = torch.abs(output)
        middleoutput = y1 + output
        fx = middleoutput
        #########################################################
        fx = fx.permute(0,2,3,1)
        # B, H, W, C = fx.shape
        fx = fx.reshape(B, -1, C)

        if self.unified_pos:
            # x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref) # 不用源代码的位置信息嵌入
            x = x.reshape(B, H*W, -1) # 用原来自带的位置编码效果就还不错的，比上面那个好
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        # else:
        #     fx = self.preprocess(x)
        #     fx = fx + self.placeholder[None, None, :]

        # if T is not None:
        #     Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
        #     Time_emb = self.time_fc(Time_emb)
        #     fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        
        fx = fx.reshape(B, H, W, self.out_dim).permute(0,3,1,2)

        return fx + up_LR, ffted_mag, ffted_mag
# from thop import profile
# def make_model(opt):
#     rgb = torch.Tensor(50, 3, 64, 64).cuda()
#     ms = torch.Tensor(50, 31, 16, 16).cuda()
#     model = Model(31,3).cuda()
#     flops, _ = profile(model, (ms, rgb, opt.sf,))
#     print('flops: ', flops/1e9)

#     total_params = sum(p.numel() for p in model.parameters())
#     print(f'{total_params/1e6:,} total parameters.')
#     total_trainable_params = sum(
#         p.numel() for p in model.parameters() if p.requires_grad)
#     print(f'{total_trainable_params/1e6:,} training parameters.')
#     return model

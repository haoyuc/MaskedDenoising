
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

class Merge_Run(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run, self).__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        init_weights(self.modules)
        
    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out

class Merge_Run_dual(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run_dual, self).__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4),
            nn.ReLU(inplace=True)
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        init_weights(self.modules)
        
    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


def init_weights(modules):
    pass
   
class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class EResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, 
                 n_channels, scale, multi_scale, 
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up =  _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, 
				 n_channels, scale, 
				 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = BasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = Merge_Run_dual(in_channels, out_channels)
        self.r2 = ResidualBlock(in_channels, out_channels)
        self.r3 = EResidualBlock(in_channels, out_channels)
        #self.g = BasicBlock(in_channels, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        
        r1 = self.r1(x)            
        r2 = self.r2(r1)       
        r3 = self.r3(r2)
        #g = self.g(r3)
        out = self.ca(r3)

        return out
        


class RIDNET(nn.Module):
    def __init__(self):
        super(RIDNET, self).__init__()
        
        n_feats = 64
        kernel_size = 3
        reduction = 16 
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = MeanShift(1, rgb_mean, rgb_std)       
        self.add_mean = MeanShift(1, rgb_mean, rgb_std, 1)

        self.head = BasicBlock(3, n_feats, kernel_size, 1, 1)

        self.b1 = Block(n_feats, n_feats)
        self.b2 = Block(n_feats, n_feats)
        self.b3 = Block(n_feats, n_feats)
        self.b4 = Block(n_feats, n_feats)

        self.tail = nn.Conv2d(n_feats, 3, kernel_size, 1, 1, 1)

    def forward(self, x):

        s = self.sub_mean(x)
        h = self.head(s)

        b1 = self.b1(h)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b_out = self.b4(b3)

        res = self.tail(b_out)
        out = self.add_mean(res)
        f_out = out + x 

        return f_out 
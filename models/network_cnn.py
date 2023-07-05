
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


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


class CNN5Layer(nn.Module):
    def __init__(self):
        super(CNN5Layer, self).__init__()
        
        n_feats = 32
        kernel_size = 3
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = MeanShift(1, rgb_mean, rgb_std)       
        self.add_mean = MeanShift(1, rgb_mean, rgb_std, 1)

        self.head = BasicBlock(3, n_feats, kernel_size, 1, 1)

        self.b1 = BasicBlock(n_feats, n_feats, kernel_size, 1, 1)
        self.b2 = BasicBlock(n_feats, n_feats, kernel_size, 1, 1)
        self.b3 = BasicBlock(n_feats, n_feats, kernel_size, 1, 1)
        self.b4 = BasicBlock(n_feats, n_feats, kernel_size, 1, 1)

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
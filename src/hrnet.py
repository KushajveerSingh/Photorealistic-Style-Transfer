import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        num_pad = int(np.floor(kernel_size / 2))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=num_pad)

    def forward(self, x):
        return self.conv(x)


class BottleneckBlock(nn.Module):
    """
    Bottleneck layer similar to resnet bottleneck layer. InstanceNorm is used
    instead of BatchNorm because when we want to generate images, we normalize
    all the images independently. 
    
    (In batch norm you compute mean and std over complete batch, while in instance 
    norm you compute mean and std for each image channel independently). The reason for 
    doing this is, the generated images are independent of each other, so we should
    not normalize them using a common statistic.
    
    If you confused about the bottleneck architecture refer to the official pytorch
    resnet implementation and paper.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        
        self.identity_block = nn.Sequential(
            ConvLayer(in_channels, out_channels//4, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_channels//4),
            nn.ReLU(),
            ConvLayer(out_channels//4, out_channels//4, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm2d(out_channels//4),
            nn.ReLU(),
            ConvLayer(out_channels//4, out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        )
        
        self.shortcut = nn.Sequential(
            ConvLayer(in_channels, out_channels, 1, stride),
            nn.InstanceNorm2d(out_channels),
        )
    
    def forward(self, x):
        out = self.identity_block(x)
        if self.in_c == self.out_c:
            residual = x
        else:
            residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out


# Helper functions for HRNet
def conv_down(in_c, out_c, stride=2):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)

def upsample(input, scale_factor):
    return F.interpolate(input=input, scale_factor=scale_factor, mode='bilinear', align_corners=False)


class HRNet(nn.Module):
    """
    For model reference see Figure 2 of the paper https://arxiv.org/pdf/1904.11617v1.pdf.
    
    Naming convention used.
    I refer to vertical layers as a single layer, so from left to right we have 8 layers
    excluding the input image.
    E.g. layer 1 contains the 500x500x16 block
         layer 2 contains 500x500x32 and 250x250x32 blocks and so on
    
    self.layer{x}_{y}:
        x :- the layer number, as explained above
        y :- the index number for that function starting from 1. So if layer 3 has two
             downsample functions I write them as `downsample3_1`, `downsample3_2`
    """
    def __init__(self):
        super().__init__()
        self.layer1_1 = BottleneckBlock(3, 16)
        
        self.layer2_1 = BottleneckBlock(16, 32)
        self.downsample2_1 = conv_down(16, 32)
        
        self.layer3_1 = BottleneckBlock(32, 32)
        self.layer3_2 = BottleneckBlock(32, 32)
        self.downsample3_1 = conv_down(32, 32)
        self.downsample3_2 = conv_down(32, 32, stride=4)
        self.downsample3_3 = conv_down(32, 32)
        
        self.layer4_1 = BottleneckBlock(64, 64)
        self.layer5_1 = BottleneckBlock(192, 64)
        self.layer6_1 = BottleneckBlock(64, 32)
        self.layer7_1 = BottleneckBlock(32, 16)
        self.layer8_1 = conv_down(16, 3, stride=1)
        
    def forward(self, x):
        map1_1 = self.layer1_1(x)
        
        map2_1 = self.layer2_1(map1_1)
        map2_2 = self.downsample2_1(map1_1)
        
        map3_1 = torch.cat((self.layer3_1(map2_1), upsample(map2_2, 2)), 1)
        map3_2 = torch.cat((self.downsample3_1(map2_1), self.layer3_2(map2_2)), 1)
        map3_3 = torch.cat((self.downsample3_2(map2_1), self.downsample3_3(map2_2)), 1)
        
        map4_1 = torch.cat((self.layer4_1(map3_1), upsample(map3_2, 2), upsample(map3_3, 4)), 1)
        
        out = self.layer5_1(map4_1)
        out = self.layer6_1(out)
        out = self.layer7_1(out)
        out = self.layer8_1(out)
        
        return out
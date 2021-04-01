# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 06:41:29 2021

@author: linhai
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
import matplotlib.pyplot as plt
#%%
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(out_channels//4, out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)
        self.gn2 = nn.GroupNorm(out_channels//4, out_channels)
        self.process_x=None    
        
        if stride > 1 and out_channels != in_channels:
            self.process_x = nn.Sequential(nn.AvgPool2d(stride,stride),
                                           nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))
        elif stride > 1 and out_channels == in_channels:
            self.process_x = nn.AvgPool2d(stride, stride)
        elif stride == 1 and out_channels != in_channels:
            self.process_x = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        #print('out', out.shape, 'x', x.shape)
        if self.process_x is not None:
            x = self.process_x(x)
        #print('out', out.shape, 'x', x.shape)
        out = out+x
        out = self.gn2(out)        
        out = self.relu2(out)
        return out
class Resnet18Unet(nn.Module):
    def __init__(self, n_seg, flag=0, offset=64, output='seg'):
        super().__init__()
        
        self.n_seg=n_seg
        self.flag=flag
        self.offset=offset
        self.output=output       
        #--------------------------------------------------------------
        self.e0 = nn.Sequential(nn.Conv2d(1, 32, 7, 2, 3),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(32, 64, 5, 2, 2, bias=False),
                                nn.GroupNorm(64, 64),
                                nn.LeakyReLU(inplace=True))
        self.e1 = nn.Sequential(Block(64, 128, 2),   Block(128, 128, 1))
        self.e2 = nn.Sequential(Block(128, 256, 2),  Block(256, 256, 1))
        self.e3 = nn.Sequential(Block(256, 512, 2),  Block(512, 512, 1))
        self.e4 = nn.Sequential(Block(512, 1024, 2), Block(1024, 1024, 1),
                                nn.Conv2d(1024, 1024, 2, 1, 0, bias=False))

        #--------------------------------------------------------------          
        
        self.g4 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1, bias=False),
                                    nn.GroupNorm(1, 512),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ConvTranspose2d(512, 512, 3, 2, 0, 0, bias=False),
                                    nn.GroupNorm(2, 512),
                                    nn.LeakyReLU(inplace=True))

        #--------------
        self.g3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1, bias=False),
                                nn.GroupNorm(4, 256),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(256, 256, 3, 2, 1, (1,0), bias=False),
                                nn.GroupNorm(4, 256),
                                nn.LeakyReLU(inplace=True))
        
        self.g2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1, bias=False),
                                nn.GroupNorm(8, 128),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(128, 128, 3, 2, 1, 1, bias=False),
                                nn.GroupNorm(8, 128),
                                nn.LeakyReLU(inplace=True))
        
        self.g1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1, bias=False),
                                nn.GroupNorm(16, 64),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(64, 64, 3, 2, 1, 1, bias=False),
                                nn.GroupNorm(16, 64),
                                nn.LeakyReLU(inplace=True))

        self.g0 = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1, bias=False),
                                nn.GroupNorm(32, 32),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(32, 32, 7, 2, 3, 1, bias=False),
                                nn.GroupNorm(32, 32),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(32, n_seg, 5, 2, 2, 1))
         
    def forward(self, x):
        #print('x', x.shape)
        x0e = self.e0(x)
        #print('x0e.shape', x0e.shape)# 40 180
        x1e = self.e1(x0e)
        #print('x1e.shape', x1e.shape)# 20 90
        x2e = self.e2(x1e)
        #print('x2e.shape', x2e.shape)# 10 45
        x3e = self.e3(x2e)
        #print('x3e.shape', x3e.shape)# 5 23
        x4e = self.e4(x3e)
        #print('x4e.shape', x4e.shape)# 2 11
        #x5e = self.e5(x4e)
        #print('x5e.shape', x5e.shape)


        x4g=x4e

        
        x4g=self.g4(x4g)
       # print('0', x4g.shape)# 5 23

        x3g=torch.cat([x3e, x4g], dim=1)
        #print('1', x3g.shape)
        x3g=self.g3(x3g)
        #print('2', x3g.shape)# 10 45

        x2g=torch.cat([x2e, x3g], dim=1)
        #print('3', x2g.shape)
        x2g=self.g2(x2g)
        #print('4', x2g.shape)#20 90

        x1g=torch.cat([x1e, x2g], dim=1)
        #print('5', x1g.shape)
        x1g=self.g1(x1g)
        #print('6', x1g.shape)# 40 180
        
        x0g=torch.cat([x0e, x1g], dim=1)
        #print('7', x0g.shape)
        x0g=self.g0(x0g)
        #print('8', x0g.shape)# 160 720
        #x0g=torch.sigmoid(x0g)
  
        return x0g
#%%


#==============================================================================================
class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=True)
        self.gn1 = nn.GroupNorm(out_channels//4, out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True)
        self.gn2 = nn.GroupNorm(out_channels//4, out_channels)
        self.process_x=None          
        
        if stride > 1 and out_channels != in_channels:
            self.process_x = nn.Sequential(nn.AvgPool2d(stride,stride, ceil_mode=True),
                                           nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True))
        elif stride > 1 and out_channels == in_channels:
            self.process_x = nn.AvgPool2d(stride, stride, ceil_mode=True)
        elif stride == 1 and out_channels != in_channels:
            self.process_x = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        #print('out', out.shape, 'x', x.shape)
        if self.process_x is not None:
            x = self.process_x(x)
        #print('out', out.shape, 'x', x.shape)
        out = out+x
        out = self.gn2(out)        
        out = self.relu2(out)
        return out
class Resnet18Unet_v2(nn.Module):
    def __init__(self, n_seg, flag=0, offset=64, output='seg'):
        super().__init__()
        
        self.n_seg=n_seg
        self.flag=flag
        self.offset=offset
        self.output=output       
        #--------------------------------------------------------------
        self.e0 = nn.Sequential(nn.Conv2d(1, 32, 7, 2, 3),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(32, 64, 5, 2, 2, bias=True),
                                nn.GroupNorm(64, 64),
                                nn.LeakyReLU(inplace=True))
        self.e1 = nn.Sequential(Block(64, 128, 2),   Block(128, 128, 1))
        self.e2 = nn.Sequential(Block(128, 256, 2),  Block(256, 256, 1))
        self.e3 = nn.Sequential(Block(256, 512, 2),  Block(512, 512, 1))
        self.e4 = nn.Sequential(Block(512, 1024, 2),  Block(1024, 1024, 1))
        self.e5 = nn.Sequential(Block(1024, 2048, 2), Block(2048, 2048, 1),
                                nn.Conv2d(2048, 2048, 2, 1, 0, bias=True))

        #--------------------------------------------------------------          
        
        self.g5 = nn.Sequential(nn.Conv2d(2048, 1024, 3, 1, 1, bias=True),
                                    nn.GroupNorm(1, 1024),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ConvTranspose2d(1024, 1024, 3, 2, 0, 1, bias=True),
                                    nn.GroupNorm(2, 1024),
                                    nn.LeakyReLU(inplace=True))

        #--------------
        
        self.g4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1, bias=True),
                                nn.GroupNorm(4, 512),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(512, 512, 3, 2, 1, 0, bias=True),
                                nn.GroupNorm(4, 512),
                                nn.LeakyReLU(inplace=True))
        
        self.g3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1, bias=True),
                                nn.GroupNorm(4, 256),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(256, 256, 3, 2, 1, 0, bias=True),
                                nn.GroupNorm(4, 256),
                                nn.LeakyReLU(inplace=True))
        
        self.g2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1, bias=True),
                                nn.GroupNorm(8, 128),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(128, 128, 3, 2, 1, 1, bias=True),
                                nn.GroupNorm(8, 128),
                                nn.LeakyReLU(inplace=True))
        
        self.g1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1, bias=True),
                                nn.GroupNorm(16, 64),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(64, 64, 3, 2, 1, 1, bias=True),
                                nn.GroupNorm(16, 64),
                                nn.LeakyReLU(inplace=True))

        self.g0 = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1, bias=True),
                                nn.GroupNorm(32, 32),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(32, 32, 7, 2, 3, 1, bias=True),
                                nn.GroupNorm(32, 32),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(32, n_seg, 5, 2, 2, 1))
         
    def forward(self, x):
        #print('x', x.shape)
        x0e = self.e0(x)
        #print('x0e.shape', x0e.shape)
        x1e = self.e1(x0e)
        #print('x1e.shape', x1e.shape)
        x2e = self.e2(x1e)
        #print('x2e.shape', x2e.shape)
        x3e = self.e3(x2e)
        #print('x3e.shape', x3e.shape)
        x4e = self.e4(x3e)
        #print('x4e.shape', x4e.shape)# 12*12
        x5e = self.e5(x4e)
        #print('x5e.shape', x5e.shape)


        x5g=x5e

        
        x5g=self.g5(x5g)
        #print('0', x5g.shape)

        x4g=torch.cat([x4e, x5g], dim=1)
        #print('1', x4g.shape)
        x4g=self.g4(x4g)
        #print('2', x3g.shape)

        x3g=torch.cat([x3e, x4g], dim=1)
        #print('1', x3g.shape)
        x3g=self.g3(x3g)
        #print('2', x3g.shape)

        x2g=torch.cat([x2e, x3g], dim=1)
        #print('3', x2g.shape)
        x2g=self.g2(x2g)
        #print('4', x2g.shape)

        x1g=torch.cat([x1e, x2g], dim=1)
        #print('5', x1g.shape)
        x1g=self.g1(x1g)
        #print('6', x1g.shape)
        
        x0g=torch.cat([x0e, x1g], dim=1)
        #print('7', x0g.shape)
        x0g=self.g0(x0g)
        #print('8', x0g.shape)
        #x0g=torch.sigmoid(x0g)
  
        return x0g
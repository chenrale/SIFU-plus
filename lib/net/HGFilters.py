# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from lib.net.net_util import *
import torch.nn as nn
import torch.nn.functional as F


class HourGlass(nn.Module):
    # lyz
    # 沙漏（HourGlass）模块，这是一种特殊的卷积神经网络结构，
    # 主要用于人体姿态估计等任务。它包含多个下采样和上采样步骤，
    # 形成一个对称的结构，形状类似沙漏
    def __init__(self, num_modules, depth, num_features, opt):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.opt = opt

        self._generate_network(self.depth)

    def _generate_network(self, level):
        # lyz
        # 递归地创建网络层。对于每个级别，它添加两个卷积块（ConvBlock，未在代码中定义）b1_ 和 b2_。
        # 如果当前级别大于1，它会递归地调用自身以创建下一级别的网络；
        # 否则，它会添加一个额外的卷积块 b2_plus_。最后，它添加一个卷积块 b3_
        self.add_module('b1_' + str(level),
                        ConvBlock(self.features, self.features, self.opt))

        self.add_module('b2_' + str(level),
                        ConvBlock(self.features, self.features, self.opt))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level),
                            ConvBlock(self.features, self.features, self.opt))

        self.add_module('b3_' + str(level),
                        ConvBlock(self.features, self.features, self.opt))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3,
                            scale_factor=2,
                            mode='bicubic',
                            align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):

    def __init__(self, opt, num_modules, in_dim):
        super(HGFilter, self).__init__()
        self.num_modules = num_modules

        self.opt = opt
        [k, s, d, p] = self.opt.conv1

        # self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(in_dim,
                               64,
                               kernel_size=k,
                               stride=s,
                               dilation=d,
                               padding=p)

        if self.opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.opt.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt)
            self.down_conv2 = nn.Conv2d(64,
                                        128,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
        elif self.opt.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt)
            self.down_conv2 = nn.Conv2d(128,
                                        128,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
        elif self.opt.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.opt)
        self.conv4 = ConvBlock(128, 256, self.opt)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module),
                            HourGlass(1, opt.num_hourglass, 256, self.opt))

            self.add_module('top_m_' + str(hg_module),
                            ConvBlock(256, 256, self.opt))
            self.add_module(
                'conv_last' + str(hg_module),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module),
                                nn.GroupNorm(32, 256))

            self.add_module(
                'l' + str(hg_module),
                nn.Conv2d(256,
                          opt.hourglass_dim,
                          kernel_size=1,
                          stride=1,
                          padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module),
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module(
                    'al' + str(hg_module),
                    nn.Conv2d(opt.hourglass_dim,
                              256,
                              kernel_size=1,
                              stride=1,
                              padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.opt.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(
                self._modules['bn_end' + str(i)](
                    self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs
    




# lyz
# 基于 hourglass 网络的深度学习模型，用于预测人体关键点
# 输入通道数in_dim、卷积核大小k、步长s、填充p
# 使用nn.BatchNorm2d或nn.GroupNorm对其进行归一化
# 定义了三个卷积层conv2、conv3和conv4，以及一个反卷积层up_conv
class FuseHGFilter(nn.Module):

    def __init__(self, opt, num_modules, in_dim):
        super(FuseHGFilter, self).__init__()
        self.num_modules = num_modules

        self.opt = opt
        [k, s, d, p] = self.opt.conv1

        # self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(in_dim,
                               64,
                               kernel_size=k,
                               stride=s,
                               dilation=d,
                               padding=p)

        if self.opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        
        self.conv2 = ConvBlock(64, 128, self.opt)
        self.down_conv2 = nn.Conv2d(128,
                                    96,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        # elif self.opt.hg_down == 'conv128':
        #     self.conv2 = ConvBlock(64, 128, self.opt)
        #     self.down_conv2 = nn.Conv2d(128,
        #                                 128,
        #                                 kernel_size=3,
        #                                 stride=2,
        #                                 padding=1)
        
        dim=96+32
        self.conv3 = ConvBlock(dim, dim, self.opt)
        self.conv4 = ConvBlock(dim, 256, self.opt)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module),
                            HourGlass(1, opt.num_hourglass, 256, self.opt))

            self.add_module('top_m_' + str(hg_module),
                            ConvBlock(256, 256, self.opt))
            self.add_module(
                'conv_last' + str(hg_module),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module),
                                nn.GroupNorm(32, 256))

            hourglass_dim=256
            self.add_module(
                'l' + str(hg_module),
                nn.Conv2d(256,
                          hourglass_dim,
                          kernel_size=1,
                          stride=1,
                          padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module),
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module(
                    'al' + str(hg_module),
                    nn.Conv2d(hourglass_dim,
                              256,
                              kernel_size=1,
                              stride=1,
                              padding=0))
                
        self.up_conv=nn.ConvTranspose2d(hourglass_dim,64,kernel_size=2,stride=2)

    def forward(self, x,plane):
        x = F.relu(self.bn1(self.conv1(x)), True)  # 64*256*256
        tmpx = x
        
        x = self.conv2(x)
        x = self.down_conv2(x)

        x=torch.cat([x,plane],1)   # 128*128*128


        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(
                self._modules['bn_end' + str(i)](
                    self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        out=self.up_conv(outputs[-1])

        return out
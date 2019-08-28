# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from pysot.core.config import cfg


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        return x
class Split_fg(nn.Module):
    def __init__(self, out_channel):
        super(Split_fg, self).__init__()
        self.split_fg = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//4, out_channel//16, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel//16),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.split_fg(x)
        return x
class Split_bg(nn.Module):
    def __init__(self, out_channel):
        super(Split_bg, self).__init__()
        self.split_bg = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//4, out_channel // 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel // 16),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.split_bg(x)
        return x

class Multi_Split_fb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Multi_Split_fb, self).__init__()
        self.num = len(in_channels)
        out_channel = out_channels[self.num-1]
        self.split_fg = Split_fg(out_channel)
        self.split_bg = Split_bg(out_channel)
        self.genmask = nn.Sequential(
            nn.Conv2d(out_channel//16, out_channel // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//16, 1, kernel_size=1, bias=False)
        )

        #self.unfold = nn.Unfold(kernel_size=(15,15),stride=4)
    #def forward(self, z_fs, x_fs, template_box):
    def forward(self, z_fs,x_fs):
        kernel = z_fs[self.num-1]
        kernel_fg = self.split_fg(kernel)
        kernel_bg = self.split_bg(kernel)

        search = x_fs[self.num-1]
        search_fg = self.split_fg(search)
        search_bg = self.split_bg(search)

        k_fmask = self.genmask(kernel_fg)
        k_bmask = self.genmask(kernel_bg)
        k_mask = torch.cat([k_fmask, k_bmask], 1)

        s_fmask = self.genmask(search_fg)
        s_bmask = self.genmask(search_bg)
        s_mask = torch.cat([s_fmask, s_bmask], 1)

        batchsize = k_mask.size(0)
        k_size = k_mask.size(2)
        s_size = s_mask.size(2)
        channels = kernel_fg.size(1)
        kfixed_noise = torch.randn(batchsize, channels, k_size, k_size).cuda()
        sfixed_noise = torch.randn(batchsize, channels, s_size, s_size).cuda()
        restruct_kimg = kernel_fg * k_mask[:, 0, :, :].unsqueeze(1) + kernel_bg * k_mask[:, 1, :, :].unsqueeze(1)
        # restruct_ksimg = search_bg * s_mask[:, 1, :, :].unsqueeze(1)
        # restruct_ksimg[:, :, 8:22, 8:22] = kernel_fg * k_mask[:, 0, :, :]
        restruct_kimg = torch.cat([restruct_kimg, kfixed_noise], 1)
        # restruct_ksimg = torch.cat([restruct_ksimg, sfixed_noise], 1)
        results={}
        results['k_mask'] = k_mask
        results['s_mask'] = s_mask
        results['k_img'] = restruct_kimg
        # results['restruct_ksimg'] = restruct_ksimg
        return results

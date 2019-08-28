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


class Split_fg(nn.Module):
    def __init__(self):
        super(Split_fg, self).__init__()
        self.split_fg = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.split_fg(x)
        return x
class Split_bg(nn.Module):
    def __init__(self):
        super(Split_bg, self).__init__()
        self.split_bg = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.split_bg(x)
        return x

class Split_fb(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Split_fb, self).__init__()
        self.split_fg = Split_fg(out_channel)
        self.split_bg = Split_bg(out_channel)
        self.adjust = nn.Conv2d(out_channel, out_channel//2, kernel_size=1, bias=False)
        self.genmask = nn.Sequential(
            nn.Conv2d(out_channel//2, out_channel // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//4, 1, kernel_size=1, bias=False)
        )
        self.unfold = nn.Unfold(kernel_size=(6,6),stride=4)
    def forward(self, batchsize, z_fs, x_fs, template_box):
        kernel_fg = split_fg(z_fs)
        kernel_bg = split_bg(z_fs)
        restruct_kfg = kernel_fg
        restruct_kbg = kernel_bg
        k_fmask = self.genmask(restruct_kfg)
        k_bmask = self.genmask(restruct_kbg)
        k_mask = torch.cat([k_fmask, k_bmask], 1)
        k_mask_fusion = F.softmax(k_mask, dim=1)
        k_batchsize = k_mask.shape[0]

        k_size = k_mask.shape[2]
        fixed_noise = torch.randn(k_batchsize, 128, k_size, k_size).cuda()
        restruct_kimg = restruct_kfg * k_mask_fusion[:, 0, :, :].view(k_batchsize, 1, 6, 6).contiguous() + \
                        restruct_kbg * k_mask_fusion[:, 1, :, :].view(k_batchsize, 1, 6, 6).contiguous()
        restruct_kimg = fixed_noise + restruct_kimg

        results={}
        results['reconstruct_kimg'] = restruct_kimg
        results['k_mask'] = k_mask

        return results

# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.models.head.rpn import DepthwiseXCorr
from pysot.core.xcorr import xcorr_depthwise


class MaskCorr(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, hidden_kernel_size=5):
        super(MaskCorr, self).__init__()
        self.generate_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels//2, in_channels//4, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, in_channels//8, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 8, in_channels // 16, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels // 32, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(in_channels // 32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 32, in_channels // 64, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm2d(in_channels // 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 64, out_channels, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 128, out_channels, kernel_size=kernel_size, stride=2, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=hidden_kernel_size, stride=1, bias=False),
            nn.Sigmoid()

        )

    def forward(self, xcorr):
        mask = self.generate_mask(xcorr)
        return mask


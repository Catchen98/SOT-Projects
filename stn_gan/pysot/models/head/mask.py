# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from pysot.models.head.rpn import DepthwiseXCorr
from pysot.core.xcorr import xcorr_depthwise


# class MaskCorr(DepthwiseXCorr):
#     def __init__(self, in_channels, hidden, out_channels,
#                  kernel_size=3, hidden_kernel_size=5):
#         super(MaskCorr, self).__init__(in_channels, hidden,
#                                        out_channels, kernel_size,
#                                        hidden_kernel_size)

#     def forward(self, kernel, search):
#         kernel = self.conv_kernel(kernel)
#         search = self.conv_search(search)
#         feature = xcorr_depthwise(search, kernel)
#         out = self.head(feature)
#         return out, feature

class MaskCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, 
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
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=hidden_kernel_size, stride=1, bias=False),
          
        )
    def forward(self, xcorr):
        mask =  self.generate_mask(xcorr)
        return mask

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v1 = nn.Sequential(
                nn.Conv2d(256, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v2 = nn.Sequential(
                nn.Conv2d(512, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h2 = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h1 = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h0 = nn.Sequential(
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)
        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, f, corr_feature, pos):
        p0 = F.pad(f[0], [16, 16, 16, 16])[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
        p1 = F.pad(f[1], [8, 8, 8, 8])[:, :, 2*pos[0]:2*pos[0]+31, 2*pos[1]:2*pos[1]+31]
        p2 = F.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0]+15, pos[1]:pos[1]+15]

        p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127*127)
        return out

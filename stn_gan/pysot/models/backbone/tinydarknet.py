from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn


class TinyDarknet(nn.Module):
    configs = [3, 16, 32, 16, 128, 16, 128, 32, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), TinyDarknet.configs))
        super(TinyDarknet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=3),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=3),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=1),
            nn.BatchNorm2d(configs[3]),
            nn.LeakyReLU(inplace=True),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.LeakyReLU(inplace=True),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=1),
            nn.BatchNorm2d(configs[5]),
            nn.LeakyReLU(inplace=True),
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(configs[5], configs[6], kernel_size=3),
            nn.BatchNorm2d(configs[6]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(configs[6], configs[7], kernel_size=1),
            nn.BatchNorm2d(configs[7]),
            nn.ReLU(inplace=True),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(configs[7], configs[8], kernel_size=3, stride=1),
            nn.BatchNorm2d(configs[8]),
        )
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x


def tinydarknet(**kwargs):
    return TinyDarknet(**kwargs)

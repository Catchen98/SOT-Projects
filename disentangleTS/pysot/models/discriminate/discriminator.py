from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from IPython import embed
from pysot.core.config import cfg

class Discriminator(nn.Module):
    def __init__(self, in_channel, middle_channel):
        super(Discriminator, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(
            # input is (nc) x 127 x 127
            nn.Conv2d(in_channel, middle_channel, 3, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 63 x 63
            nn.Conv2d(middle_channel, middle_channel * 2, 3, 2, bias=False),
            nn.BatchNorm2d(middle_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 31 x 31
            nn.Conv2d(middle_channel * 2, middle_channel * 4, 3, 2, bias=False),
            nn.BatchNorm2d(middle_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 15 x 15
            nn.Conv2d(middle_channel * 4, middle_channel * 8, 3, 2, bias=False),
            nn.BatchNorm2d(middle_channel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 5*5
            nn.Conv2d(middle_channel * 8, middle_channel * 8, 3, 2, bias=False),
            nn.BatchNorm2d(middle_channel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 3 x 3
            nn.Conv2d(middle_channel * 8, 1, 3, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):

        output = self.main(input)
        
        return output.view(-1, 1).squeeze(1)


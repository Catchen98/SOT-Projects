# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/jn/codes/Deformablev2')
# from pysot.models.deformable.modules import DeformConv
from modules import DeformConv
from IPython import embed

class DeformConvNet(nn.Module):
    def __init__(self,in_channel,out_channel):
        num_deformable_groups = 1
        super(DeformConvNet,self).__init__()
        self.Toffset=nn.Conv2d(in_channel,num_deformable_groups*2*3*3,kernel_size=3,padding=1)
        self.Tdeform=DeformConv(in_channel,out_channel,kernel_size=3,padding=1,
                                num_deformable_groups=num_deformable_groups)
        self.Soffset=nn.Conv2d(in_channel,num_deformable_groups*2*3*3,kernel_size=3,padding=1)
        self.Sdeform=DeformConv(in_channel,out_channel,kernel_size=3,padding=1,
                                num_deformable_groups=num_deformable_groups)
    def forward(self,kernel,search):
        kernel_offset=self.Toffset(kernel)
        kernel=self.Tdeform(kernel,kernel_offset)
        search_offset=self.Soffset(search)
        search=self.Sdeform(search,search_offset)
        outputs={}
        outputs['kernel']=kernel
        outputs['search']=search
        outputs['kernel_offset']=kernel_offset
        outputs['search_offset']=search_offset
        return outputs
            

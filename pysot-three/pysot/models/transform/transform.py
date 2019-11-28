# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================
**Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_
.. figure:: /_static/img/stn/FSeq.png
In this tutorial, you will learn how to augment your network using
a visual attention mechanism called spatial transformer
networks. You can read more about the spatial transformer
networks in the `DeepMind paper <https://arxiv.org/abs/1506.02025>`__
Spatial transformer networks are a generalization of differentiable
attention to any spatial transformation. Spatial transformer networks
(STN for short) allow a neural network to learn how to perform spatial
transformations on the input image in order to enhance the geometric
invariance of the model.
For example, it can crop a region of interest, scale and correct
the orientation of an image. It can be a useful mechanism because CNNs
are not invariant to rotation and scale and more general affine
transformations.
One of the best things about STN is the ability to simply plug it into
any existing CNN with very little modification.
"""
# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from pysot.models.manytools import get_meshgrid


class T_Net(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(T_Net, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_channel,in_channel//2,kernel_size=3),#4
            nn.BatchNorm2d(in_channel//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//2,in_channel//4,kernel_size=3,padding=1),#4
            nn.BatchNorm2d(in_channel//4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//4,in_channel//8,kernel_size=3,padding=1),#4
            nn.BatchNorm2d(in_channel//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//8,in_channel//16,kernel_size=3,padding=1),#4
            nn.BatchNorm2d(in_channel//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//16,in_channel//32,kernel_size=3,padding=1),#8
            nn.BatchNorm2d(in_channel//32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//32,in_channel//32,kernel_size=3,padding=1),#6
            nn.BatchNorm2d(in_channel//32),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channel//32,in_channel//64,kernel_size=3),#6
            nn.BatchNorm2d(in_channel//64),
            nn.LeakyReLU(inplace=True),
            # nn.ConvTranspose2d(in_channel//64,out_channel*8,kernel_size=3),#10
            # nn.BatchNorm2d(out_channel*8),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(out_channel*8,out_channel*4,kernel_size=3),#12
            # nn.BatchNorm2d(out_channel*4),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(out_channel*4,out_channel*2,kernel_size=3),#14
            # nn.BatchNorm2d(out_channel*2),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(out_channel*2,out_channel*2,kernel_size=3),#16
            # nn.BatchNorm2d(out_channel*2),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channel*4,out_channel,kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            # nn.Sigmoid()
        )
        # for m in self.modules():
        #   if isinstance(m,nn.Conv2d):
        #     nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='ReLU')
        #   elif isinstance(m, nn.GroupNorm):
        #     nn.init.constant_(m.weight,1)
        #     nn.init.constant_(m.bias,0)
    def forward(self, input):
        
        matrix=self.transform(input)
        
        return matrix
class Transform(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Transform, self).__init__()
        self.transform=T_Net(in_channel,out_channel)
    def forward(self,zf,xf):
        batch_size=zf.shape[0]
        self.size=zf.shape[-1]
        self.mesh_grid=get_meshgrid(batch_size,self.size,self.size)
        input1=torch.cat([zf,xf],dim=1)
        input2=torch.cat([xf,zf],dim=1)
        matrix1=self.transform(input1)
        matrix2=self.transform(input2)
        v_grid1=self.get_vgrid(matrix1)
        v_grid2=self.get_vgrid(matrix2)
        warped_zfs=F.grid_sample(input1,v_grid1)
        warped_xfs=F.grid_sample(input2,v_grid2)
        warped_zf=warped_zfs[:,0:256,:,:]
        warped_xf=warped_xfs[:,0:256,:,:]
        outputs={}
        outputs['zx_matrix']=matrix1
        outputs['xz_matrix']=matrix2
        outputs['warped_zf']=warped_zf
        outputs['warped_xf']=warped_xf
        outputs['zx_grid']=v_grid1
        outputs['xz_grid']=v_grid2
        return outputs
    def get_vgrid(self, flow):
        
        vgrid = self.mesh_grid + flow
        # make the range of vgrid-value in (-1,1), grid_sample
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(self.size - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(self.size - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        return vgrid

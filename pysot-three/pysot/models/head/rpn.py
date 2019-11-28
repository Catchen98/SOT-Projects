# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import numpy as np
from pysot.core.xcorr import xcorr_fast, xcorr_depthwise
from pysot.models.init_weight import init_weights
from pysot.core.config import cfg
from pysot.models.commonsense import get_commonsense
from pysot.models.deformable import get_deformable
from pysot.models.transform import get_transform
from pysot.models.manytools import get_meshgrid


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, feature_in=256):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        if cfg.COMMONSENSE.COMMONSENSE:
            self.commonsense=get_commonsense(cfg.COMMONSENSE.TYPE)
            if cfg.COMMONSENSE.TYPE=='Explicit_corr':
                self.commonhead=nn.Sequential(
                    nn.Conv2d(hidden+1, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, out_channels, kernel_size=1)
                    )
            if cfg.COMMONSENSE.TYPE=='Explicit_corr1':
                self.commonhead=nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, out_channels, kernel_size=1)
                    )
            if cfg.COMMONSENSE.TYPE=='Explicit_corr2':
                self.commonhead=nn.Sequential(
                    nn.Conv2d(hidden+16, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, out_channels, kernel_size=1)
                    )
            
        if cfg.DEFORMABLE.DEFORMABLE:
            self.deform_layer=get_deformable(cfg.DEFORMABLE.TYPE,
                                             **cfg.DEFORMABLE.KWARGS)
        if cfg.TSF.TSF:
            self.tsf = get_transform(cfg.TSF.TYPE,
                                         **cfg.TSF.KWARGS)
    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        # embed()
        if cfg.COMMONSENSE.COMMONSENSE:
            original_xcorr = xcorr_depthwise(search, kernel)
            original_xcorr = self.head(original_xcorr)
            Sf=kernel
            common_outputs=self.commonsense(kernel,search)
            common_kernel=common_outputs['common_template']
            common_search=common_outputs['common_search']
            common_kernel=common_kernel.unsqueeze(4).unsqueeze(5)
            common_kernel=common_kernel.expand_as(common_search)
            # feature = xcorr_depthwise(search, kernel)
            feature=torch.sum(torch.sum(common_kernel*common_search,dim=2),dim=2)
            xcorr = self.commonhead(feature)
            # embed()
            outputs={}
            outputs['xcorr']=xcorr
            outputs['original_xcorr']=original_xcorr
            outputs['ksf']=Sf
            outputs['kcf']=common_outputs['common_template']
            outputs['scf']=common_search
            outputs['ssf']=search
            outputs['kernel_matrix']=common_outputs['template_weight']
            outputs['search_matrix']=common_outputs['search_weight']
            return outputs
        if cfg.DEFORMABLE.DEFORMABLE:
            original_xcorr = xcorr_depthwise(search, kernel)
            original_xcorr = self.head(original_xcorr)
            Sf=kernel
            deform_outputs=self.deform_layer(kernel,search)
            deform_kernel=deform_outputs['kernel']
            deform_search=deform_outputs['search']
            Cf=deform_kernel
            feature = xcorr_depthwise(deform_search, deform_kernel)
            out = self.head(feature)
            
            outputs={}
            outputs['xcorr']=out
            outputs['original_xcorr']=original_xcorr
            outputs['ksf']=Sf
            outputs['kcf']=Cf
            outputs['scf']=deform_search
            outputs['ssf']=search
            outputs['kernel_matrix']=deform_outputs['kernel_offset']
            outputs['search_matrix']=deform_outputs['search_offset']
            return outputs
        if cfg.TSF.TSF:
            # feature=xcorr_depthwise(search,kernel)
            # out=self.head(feature)
            batch_size=cfg.TRAIN.BATCH_SIZE
            size=kernel.shape[2]
            channel=kernel.shape[1]
            # xcorr=xcorr_depthwise(search, kernel)
            # xcorr = torch.sum(xcorr,dim=1).unsqueeze(1)
            # xcorr -= torch.min(xcorr)
            # xcorr /= torch.max(xcorr)
            # score=xcorr.reshape(batch_size,-1)
            # nums = torch.argmax(score,dim=1)
            # warped_zfs=kernel.clone()
            # warped_xcorrzf=kernel.clone()
            # warped_xfs=search.clone()
            # for i in range(0,len(nums)):
            #     pos = np.unravel_index(nums[i].detach().cpu().numpy(),(1,xcorr.shape[1],xcorr.shape[2],xcorr.shape[3]))
                # tsf_outputs=self.tsf(kernel[i,:,:,:].unsqueeze(0),
                #                                             search[i,:,pos[2]:pos[2]+size,pos[3]:pos[3]+size].unsqueeze(0))
        
            #     warped_xcorrzf[i,:,:,:]=tsf_outputs['warped_zf'].squeeze()
            #     warped_xfs[i,:,pos[2]:pos[2]+size,pos[3]:pos[3]+size]=tsf_outputs['warped_zf'].squeeze()
            #     warped_zfs[i,:,:,:]=tsf_outputs['warped_xf'].squeeze()
            #     if i==0:
            #         zx_transform=tsf_outputs['zx_matrix']
            #         xz_transform=tsf_outputs['xz_matrix']
            #         zx_grid=tsf_outputs['zx_grid']
            #         xz_grid=tsf_outputs['xz_grid']
            #         vis_ztsf=tsf_outputs['warped_zf']
            #         vis_xtsf=tsf_outputs['warped_xf']
            #         vis_zf=kernel[i,:,:,:]
            #         vis_xf=search[i,:,pos[2]:pos[2]+size,pos[3]:pos[3]+size]
            #     else:
            #         zx_transform=torch.cat([zx_transform,tsf_outputs['zx_matrix']],dim=0)
            #         xz_transform=torch.cat([xz_transform,tsf_outputs['xz_matrix']],dim=0)
            #         zx_grid=torch.cat([zx_grid,tsf_outputs['zx_grid']],dim=0)
            #         xz_grid=torch.cat([xz_grid,tsf_outputs['xz_grid']],dim=0)
            # warped_feature = xcorr_depthwise(search,warped_xcorrzf)
            # warped_xcorr=self.head(warped_feature)
            tsf_outputs=self.tsf(kernel,search)
            zx_transform=tsf_outputs['zx_matrix']
            xz_transform=tsf_outputs['xz_matrix']
            zx_grid=tsf_outputs['zx_grid']
            xz_grid=tsf_outputs['xz_grid']
            vis_ztsf=tsf_outputs['warped_zf'][0,:,:,:]
            vis_xtsf=tsf_outputs['warped_xf'][0,:,:,:]
            warped_xfs=tsf_outputs['warped_zf']
            warped_zfs=tsf_outputs['warped_xf']
            outputs={}
            # outputs['warped_xcorr']=warped_xcorr
            # outputs['original_xcorr']=out
            outputs['vis_ztsf']=vis_ztsf.unsqueeze(0)
            outputs['vis_xtsf']=vis_xtsf.unsqueeze(0)
            # outputs['vis_zf']=vis_zf
            # outputs['vis_xf']=vis_xf
            outputs['warped_zfs']=warped_zfs
            outputs['warped_xfs']=warped_xfs
            outputs['zx_transform']=zx_transform
            outputs['xz_transform']=xz_transform
            outputs['zx_grid']=zx_grid
            outputs['xz_grid']=xz_grid
            outputs['xf']=search
            outputs['zf']=kernel
            return outputs
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        outputs={}
        outputs['xcorr']=out
        outputs['xf']=search
        outputs['zf']=kernel
        return outputs
        
        


class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        if cfg.COMMONSENSE.COMMONSENSE or cfg.DEFORMABLE.DEFORMABLE:
            cls = self.cls(z_f, x_f)
            loc = self.loc(z_f, x_f)
            return cls, loc
        if cfg.TSF.TSF:
            cls = self.cls(z_f,x_f)
            loc = self.loc(z_f,x_f)
            return cls,loc
        cls=self.cls(z_f,x_f)
        loc=self.loc(z_f,x_f)
        return cls,loc


class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)

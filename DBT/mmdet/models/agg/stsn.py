from IPython import embed
import torch.nn as nn
import logging
import torch
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import AGG,BACKBONES
from ..utils import build_conv_layer, build_norm_layer
import numpy as np

@AGG.register_module
class STSN(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27

        self.conv1_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv1 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv2_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv3_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv3 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv4_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv4 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.relu=nn.LeakyReLU(inplace=True)
        self.norm1 = nn.GroupNorm(32,in_channels)
        self.norm2 = nn.GroupNorm(32,in_channels)
        self.norm3 = nn.GroupNorm(32,in_channels)
        self.norm4 = nn.GroupNorm(32,out_channels)
        # self.similarity=nn.Sequential(
        #     build_conv_layer(None, 512, 256, kernel_size=1, stride=1, padding=0,bias=False),
        #     nn.GroupNorm(32,256),
        #     nn.LeakyReLU(inplace=True),
        #     build_conv_layer(None, 256, 128, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.GroupNorm(32,128),
        #     nn.LeakyReLU(inplace=True),
        #     build_conv_layer(None, 128, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.GroupNorm(32,64),
        #     nn.LeakyReLU(inplace=True),
        #     build_conv_layer(None, 64, 1, kernel_size=1, stride=1, padding=0, bias=False),
        #     )
        self.neck=nn.Sequential(
            build_conv_layer(None, out_channels, 512, kernel_size=1, stride=1, padding=0,bias=False),
            nn.GroupNorm(32,512),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32,256),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 256, 512, kernel_size=1, stride=1, padding=0, bias=False))
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def agg(self,support,reference):
        features=torch.cat([support,reference],dim=1)
        # print(features.shape)
        offset1=self.conv1_offset(features)
        # self.offset1=offset1
        agg_features1=self.conv1(features,offset1)
        agg_features1=self.norm1(agg_features1)
        agg_features1=self.relu(agg_features1)
        # print('agg1_nan',torch.isnan(agg_features1).any())
        # print('agg1_inf',torch.isinf(agg_features1).any())
        offset2=self.conv2_offset(agg_features1)
        # self.offset2=offset2
        agg_features2=self.conv2(agg_features1,offset2)
        agg_features2=self.norm2(agg_features2)
        agg_features2=self.relu(agg_features2)
        # print('agg2_nan',torch.isnan(agg_features2).any())
        # print('agg2_inf',torch.isinf(agg_features2).any())
        offset3=self.conv3_offset(agg_features2)
        # self.offset3=offset3
        agg_features3=self.conv3(agg_features2,offset3)
        agg_features3=self.norm3(agg_features3)
        agg_features3=self.relu(agg_features3)
        # print('agg3_nan',torch.isnan(agg_features3).any())
        # print('agg3_inf',torch.isinf(agg_features3).any())
        offset4=self.conv4_offset(agg_features3)
        # print('offset4_nan',torch.isnan(offset4).any())
        # print('support',torch.isnan(support).any())
        # self.offset4=offset4
        agg_features=self.conv4(support,offset4)
        # print('agg_f4_nan',torch.isnan(agg_features).any())
        # print('agg_f4_inf',torch.isinf(agg_features).any())
        # print(agg_features)
        agg_features=self.norm4(agg_features)
        # print('agg_f4_2',torch.isnan(agg_features).any())
        agg_features=self.relu(agg_features)
        # print('agg_f4_3',torch.isnan(agg_features).any())
        # print('agg4',agg_features)
        return agg_features

    def forward(self,datas):
        # split=datas.shape[0]//2
        # reference=datas[:split,:,:,:]+0
        # support=datas[split:,:,:,:]+0
        reference=datas+0
        shuffle_id=np.random.randint(low=0,high=reference.shape[0],size=reference.shape[0])
        support=datas[shuffle_id,:,:,:]+0
        
        tt_feature=self.agg(reference,reference)
        stt=self.neck(tt_feature)
        ttweight=torch.exp(torch.cosine_similarity(stt,stt,dim=1).unsqueeze(1))
        
        tk_feature=self.agg(support,reference)
        stk=self.neck(tk_feature)
        tkweight=torch.exp(torch.cosine_similarity(stt,stk,dim=1).unsqueeze(1))
        
        weights=torch.cat([ttweight.unsqueeze(0),tkweight.unsqueeze(0)],dim=0)#(2,b,1,w,h)
        weights=F.softmax(weights,dim=0)
        
        features=torch.cat([tt_feature.unsqueeze(0),tk_feature.unsqueeze(0)],dim=0)#(2,b,c,w,h)
        agg_features=torch.sum(weights*features,dim=0)#(b,c,w,h)
        
        return agg_features
    def stsn(self,support,reference):
        
        tt_feature=self.agg(reference,reference)
        stt=self.neck(tt_feature)
        
        # print(ttweight.max(),ttweight.min())
        tk_feature=self.agg(support,reference)
        stk=self.neck(tk_feature)
        # stk=tk_feature
        tkweight=torch.exp(torch.cosine_similarity(stt,stk,dim=1).unsqueeze(1))
        # print(tkweight.max(),tkweight.min())
        
        return tkweight
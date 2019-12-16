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
        self.relu=nn.ReLU(inplace=True)
        self.norm1_name, norm1 = build_norm_layer(dict(type='GN',num_groups=32), in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(dict(type='GN',num_groups=32), in_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(dict(type='GN',num_groups=32), in_channels, postfix=3)
        self.norm4_name, norm4 = build_norm_layer(dict(type='GN',num_groups=32), out_channels, postfix=4)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        self.add_module(self.norm3_name, norm3)
        self.add_module(self.norm4_name, norm4)
        self.similarity=nn.Sequential(
            build_conv_layer(None, 256, 512, kernel_size=1, stride=1, bias=False),
            build_conv_layer(None, 512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            build_conv_layer(None, 512, 256, kernel_size=1, stride=1, bias=False))
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
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)
    @property
    def norm4(self):
        return getattr(self, self.norm4_name)   
    def agg(self,support,reference):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features1=self.conv1(features,offset1)
        agg_features1=self.norm1(agg_features1)
        agg_features1=self.relu(agg_features1)
        offset2=self.conv2_offset(agg_features1)
        agg_features2=self.conv2(agg_features1,offset2)
        agg_features2=self.norm2(agg_features2)
        agg_features2=self.relu(agg_features2)
        offset3=self.conv3_offset(agg_features2)
        agg_features3=self.conv3(agg_features2,offset3)
        agg_features3=self.norm3(agg_features3)
        agg_features3=self.relu(agg_features3)
        offset4=self.conv4_offset(agg_features3)
        agg_features=self.conv4(support,offset4)
        agg_features=self.norm4(agg_features)
        agg_features=self.relu(agg_features)
        return agg_features
    def forward(self,datas):
        # embed()
        support=datas[:1,:,:,:].clone()
        reference=datas[1:,:,:,:].clone()
        tt_feature=self.agg(reference,reference)
        # stt=self.similarity(tt_feature)
        stt=tt_feature
        ttweight=torch.cosine_similarity(stt,stt,dim=1).unsqueeze(1)#(b,1,w,h)
        # print(ttweight.max(),ttweight.min())
        tk_feature=self.agg(support,reference)
        # stk=self.similarity(tk_feature)
        stk=tk_feature
        tkweight=torch.cosine_similarity(stt,stk,dim=1).unsqueeze(1)
        # print(tkweight.max(),tkweight.min())
        weights=torch.cat([ttweight.unsqueeze(0),tkweight.unsqueeze(0)],dim=0)#(2,b,1,w,h)
        weights=F.softmax(weights,dim=0)
        
        features=torch.cat([tt_feature.unsqueeze(0),tk_feature.unsqueeze(0)],dim=0)#(2,b,c,w,h)
        agg_features=torch.sum(weights*features,dim=0)#(b,c,w,h)
        # print(weights)
        return agg_features
        # return stt
    def test_stsn(self,support,reference):
        
        tt_feature=self.agg(reference,reference)
        stt=self.similarity(tt_feature)
        
        # print(ttweight.max(),ttweight.min())
        tk_feature=self.agg(support,reference)
        stk=self.similarity(tk_feature)
        # stk=tk_feature
        tkweight=torch.cosine_similarity(stt,stk,dim=1).unsqueeze(1)
        # print(tkweight.max(),tkweight.min())
        
        return tkweight
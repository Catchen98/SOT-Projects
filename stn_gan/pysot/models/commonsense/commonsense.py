# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2019-05-17 20:39:34
# @Last Modified by:   yulidong
# @Last Modified time: 2019-05-18 00:36:39
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
class similarity_measure1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(similarity_measure1, self).__init__()
        self.inplanes = 32
        self.commonsense=nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1, padding=0,
                        bias=False,dilation=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel//2, kernel_size=1, stride=1, padding=0,
                        bias=False,dilation=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//2, in_channel//4, kernel_size=1, stride=1, padding=0,
                        bias=False,dilation=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//4, in_channel//8, kernel_size=1, stride=1, padding=0,
                        bias=False,dilation=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//8, in_channel//16, kernel_size=1, stride=1, padding=0,
                        bias=False,dilation=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//16, in_channel//32, kernel_size=1, stride=1, padding=0,
                        bias=False,dilation=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel//32, 1, kernel_size=1, stride=1, padding=0,
                        bias=False,dilation=1),
        )
        
        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
          elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def forward(self, x):

        output=self.commonsense(x)
        return output


class Explicit_corr(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Explicit_corr,self).__init__()
        #here you can simply use the normalized l2 loss or cosine
        self.similarity1=similarity_measure1(in_channel,out_channel)
        self.sigmoid=nn.Sigmoid()
        #self.distance_matrix=matrix_generation()
        #7 is the size of template
        self.unfold = nn.Unfold(kernel_size=(15,15))
    def forward(self, template_feature,candidate_feature):
        
        size=template_feature.shape[2]
        template_feature_mean = torch.sum(torch.sum(template_feature,dim=-1,keepdim=True),dim=-2,keepdim=True).expand_as(template_feature) \
                              /(template_feature.shape[-1]*template_feature.shape[-2])#######(N,256,6,6)

        representation = torch.cat([template_feature_mean, template_feature],1).float()########(N,256*2,6,6)

        weight=self.similarity1(representation)######(N,1,6,6)
        weight_norm=F.softmax(weight.view(weight.shape[0],1,-1,1), dim=2).view_as(weight)########(N,1,6,6)
        weight_norm=weight_norm.reshape(weight_norm.shape[0],1,size,size)
            # if idx==1:
            #    print(weight_norm)
        common_template=torch.sum(torch.sum(template_feature*weight_norm,dim=-1,keepdim=True),dim=-2,keepdim=True)########(N,256,1,1)
            #representation = torch.cat([common_template.expand_as(template_feature),template_feature],1).float()
        
        #candidate feature
        candidate_feature_all=self.unfold(candidate_feature).view(candidate_feature.shape[0],candidate_feature.shape[1],size,size,17,17).contiguous()
        candidate_feature_mean=torch.sum(torch.sum(candidate_feature_all,dim=2,keepdim=True),dim=3,keepdim=True).expand_as(candidate_feature_all) \
                                /(template_feature.shape[-1]*template_feature.shape[-2])

        representation_c = torch.cat([candidate_feature_mean, candidate_feature_all],1)  #######(N,256*2,6,6,17,17)
        representation_c=representation_c.permute(0,4,5,1,2,3).contiguous().view(representation_c.shape[0]*17*17,representation_c.shape[1],size,size).contiguous()
        weight_c=self.similarity1(representation_c)######(N*17*17,1,6,6)
        weight_c_norm=F.softmax(weight_c.view(weight_c.shape[0],1,-1,1), dim=2).view(representation_c.shape[0]//17//17,17,17,1,size,size).permute(0,3,4,5,1,2).contiguous()
        common_candidate=torch.sum(torch.sum(candidate_feature_all*weight_c_norm,dim=2,keepdim=True),dim=3,keepdim=True)#####(N,1,17,17)
        #representation_c = torch.cat([common_candidate.expand_as(candidate_feature_all),candidate_feature_all],1)
        #representation_c = representation_c.permute(0,4,5,1,2,3).contiguous().view(representation_c.shape[0]*17*17,representation_c.shape[1],size,size).contiguous()
        return  common_template.view(common_template.shape[0],common_template.shape[1],1,1),common_candidate.view(common_candidate.shape[0],common_candidate.shape[1],17,17)
    def xcorr_depthwise(self,common_candidate,common_template):
        """depthwise cross correlation
        """
        batch = common_template.size(0)
        channel = common_template.size(1)
        common_candidate = common_candidate.view(1, batch*channel, common_candidate.size(2), common_candidate.size(3))
        common_template = common_template.view(batch*channel, 1, common_template.size(2), common_template.size(3))
        out = F.conv2d(common_candidate, common_template, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

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
    def __init__(self):
        super(similarity_measure1, self).__init__()
        self.inplanes = 32
        self.conv0 = nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu0 = nn.LeakyReLU(inplace=True)        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)        
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
          elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def forward(self, x):

        output = self.conv0(x)
        output = self.relu0(output)
        output = self.conv1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        return output



def matrix_generation():
    #scale is the size of template
    scale=6
    x=torch.arange(-(scale//2),scale//2).float().unsqueeze(0)
    distance_matrix=x.expand(scale,scale).unsqueeze(0)
    distance_matrix=torch.cat([distance_matrix,distance_matrix.transpose(2,1)],0)
    distance_matrix=distance_matrix.view(1,distance_matrix.size(0),scale,scale).unsqueeze(0)
    return distance_matrix.cuda()
# contact weight
class Explicit_corr(nn.Module):
    def __init__(self):
        super(Explicit_corr,self).__init__()
        #here you can simply use the normalized l2 loss or cosine
        self.similarity1=similarity_measure1()
        self.sigmoid=nn.Sigmoid()
        #self.distance_matrix=matrix_generation()
        #7 is the size of template
        self.unfold = nn.Unfold(kernel_size=(4,4))
    def forward(self, template_feature, candidate_feature):
        size=template_feature.shape[2]
        template_feature_mean=torch.sum(torch.sum(template_feature,dim=-1,keepdim=True),dim=-2,keepdim=True).expand_as(template_feature) \
                              /(template_feature.shape[-1]*template_feature.shape[-2])#######(N,256,6,6)
        # location_matrix = matrix_generation()
        # representation = torch.cat([template_feature_mean, template_feature],1).float()########(N,256*2+2,6,6)

        # weight=self.similarity1(representation)######(N,1,6,6)
        weight = F.cosine_similarity(template_feature_mean, template_feature,dim=1)
        weight_norm=F.softmax(weight.view(weight.shape[0],1,-1,1), dim=2).view_as(weight)########(N,1,6,6)
        weight_norm=weight_norm.reshape(weight_norm.shape[0],1,size,size)
            # if idx==1:
            #    print(weight_norm)
        # common_template=torch.sum(torch.sum(template_feature*weight_norm,dim=-1,keepdim=True),dim=-2,keepdim=True)########(N,256,1,1)
        common_template=torch.cat([template_feature,weight_norm],dim=1)
            #representation = torch.cat([common_template.expand_as(template_feature),template_feature],1).float()
        
        #candidate feature
        candidate_feature_all=self.unfold(candidate_feature).view(candidate_feature.shape[0],candidate_feature.shape[1],size,size,17,17).contiguous()
        candidate_feature_mean=torch.sum(torch.sum(candidate_feature_all,dim=2,keepdim=True),dim=3,keepdim=True).expand_as(candidate_feature_all) \
                                /(template_feature.shape[-1]*template_feature.shape[-2])

        # representation_c = torch.cat([candidate_feature_mean, candidate_feature_all],1)  #######(N,256*2,6,6,17,17)
        # representation_c=representation_c.permute(0,4,5,1,2,3).contiguous().view(representation_c.shape[0]*17*17,representation_c.shape[1],size,size).contiguous()
        # weight_c=self.similarity1(representation_c)######(N*17*17,1,6,6)
        weight_c=F.cosine_similarity(candidate_feature_mean, candidate_feature_all,dim=1)
        weight_c_norm=F.softmax(weight_c.view(weight_c.shape[0],1,-1,1), dim=2).view(template_feature.shape[0],17,17,1,size,size).permute(0,3,4,5,1,2).contiguous()
        # common_candidate=torch.sum(torch.sum(candidate_feature_all*weight_c_norm,dim=2,keepdim=True),dim=3,keepdim=True)#####(N,1,17,17)
        common_candidate=torch.cat([candidate_feature_all,weight_c_norm],dim=1)
        #representation_c = torch.cat([common_candidate.expand_as(candidate_feature_all),candidate_feature_all],1)
        #representation_c = representation_c.permute(0,4,5,1,2,3).contiguous().view(representation_c.shape[0]*17*17,representation_c.shape[1],size,size).contiguous()
        # return common_template.view(common_template.shape[0],common_template.shape[1],1,1),common_candidate.view(common_candidate.shape[0],common_candidate.shape[1],17,17)
        # embed()
        # exit()
        return {
            'common_template':common_template,
            'common_search':common_candidate,
            'template_weight':weight_norm,
            'search_weight':weight_c_norm,
        }
# fusion
class Explicit_corr1(nn.Module):
    def __init__(self):
        super(Explicit_corr1,self).__init__()
        #here you can simply use the normalized l2 loss or cosine
        self.similarity1=similarity_measure1()
        self.sigmoid=nn.Sigmoid()
        #self.distance_matrix=matrix_generation()
        #7 is the size of template
        self.unfold = nn.Unfold(kernel_size=(4,4))
    def forward(self, template_feature, candidate_feature):
        size=template_feature.shape[2]
        common_template=template_feature.clone()
        for i in range(3):
            template_feature_mean=torch.sum(torch.sum(common_template,dim=-1,keepdim=True),dim=-2,keepdim=True).expand_as(common_template) \
                                /(template_feature.shape[-1]*template_feature.shape[-2])#######(N,256,6,6)
            # location_matrix = matrix_generation()
            representation = torch.cat([template_feature_mean, common_template],1).float()########(N,256*2+2,6,6)
            
            weight=self.similarity1(representation)######(N,1,6,6)
            weight_norm=F.softmax(weight.view(weight.shape[0],1,-1,1), dim=2).view_as(weight)########(N,1,6,6)
            weight_norm=weight_norm.reshape(weight_norm.shape[0],1,size,size)
            common_template=common_template*weight_norm
        common_template=torch.sum(torch.sum(common_template,dim=-1,keepdim=True),dim=-2,keepdim=True)########(N,256,1,1)
            #representation = torch.cat([common_template.expand_as(template_feature),template_feature],1).float()
        
        #candidate feature
        candidate_feature_all=self.unfold(candidate_feature).view(candidate_feature.shape[0],candidate_feature.shape[1],size,size,17,17).contiguous()
        for i in range(3):
            candidate_feature_mean=torch.sum(torch.sum(candidate_feature_all,dim=2,keepdim=True),dim=3,keepdim=True).expand_as(candidate_feature_all) \
                                    /(template_feature.shape[-1]*template_feature.shape[-2])
            
            representation_c = torch.cat([candidate_feature_mean, candidate_feature_all],1)  #######(N,256*2,6,6,17,17)
            representation_c=representation_c.permute(0,4,5,1,2,3).contiguous().view(representation_c.shape[0]*17*17,representation_c.shape[1],size,size).contiguous()
            weight_c=self.similarity1(representation_c)######(N*17*17,1,6,6)
            weight_c_norm=F.softmax(weight_c.view(weight_c.shape[0],1,-1,1), dim=2).view(representation_c.shape[0]//17//17,17,17,1,size,size).permute(0,3,4,5,1,2).contiguous()
            candidate_feature_all=candidate_feature_all*weight_c_norm
        common_candidate=torch.sum(torch.sum(candidate_feature_all,dim=2,keepdim=True),dim=3,keepdim=True)#####(N,1,17,17)
        #representation_c = torch.cat([common_candidate.expand_as(candidate_feature_all),candidate_feature_all],1)
        #representation_c = representation_c.permute(0,4,5,1,2,3).contiguous().view(representation_c.shape[0]*17*17,representation_c.shape[1],size,size).contiguous()
        # return common_template.view(common_template.shape[0],common_template.shape[1],1,1),common_candidate.view(common_candidate.shape[0],common_candidate.shape[1],17,17)
        return {
            'common_template':common_template,
            'common_search':common_candidate,
            'template_weight':weight_norm,
            'search_weight':weight_c_norm,
        }
# fusion and contact
class Explicit_corr2(nn.Module):
    def __init__(self):
        super(Explicit_corr2,self).__init__()
        #here you can simply use the normalized l2 loss or cosine
        self.similarity1=similarity_measure1()
        self.sigmoid=nn.Sigmoid()
        #self.distance_matrix=matrix_generation()
        #7 is the size of template
        self.unfold = nn.Unfold(kernel_size=(4,4))
    def forward(self, template_feature, candidate_feature):
        size=template_feature.shape[2]
        template_feature_mean=torch.sum(torch.sum(template_feature,dim=-1,keepdim=True),dim=-2,keepdim=True).expand_as(template_feature) \
                            /(template_feature.shape[-1]*template_feature.shape[-2])#######(N,256,6,6)
        # location_matrix = matrix_generation()
        representation = torch.cat([template_feature_mean, template_feature],1).float()########(N,256*2+2,6,6)

        weight=self.similarity1(representation)######(N,1,6,6)
        weight_norm=F.softmax(weight.view(weight.shape[0],1,-1,1), dim=2).view_as(weight)########(N,1,6,6)
        weight_norm=weight_norm.reshape(weight_norm.shape[0],1,size,size)
            # if idx==1:
            #    print(weight_norm)
        common_template=torch.sum(torch.sum(template_feature*weight_norm,dim=-1,keepdim=True),dim=-2,keepdim=True)########(N,256,1,1)
            #representation = torch.cat([common_template.expand_as(template_feature),template_feature],1).float()
        common_template=torch.cat([common_template,weight_norm.reshape(common_template.shape[0],-1,1,1)],dim=1)
        #candidate feature
        candidate_feature_all=self.unfold(candidate_feature).view(candidate_feature.shape[0],candidate_feature.shape[1],size,size,17,17).contiguous()
        candidate_feature_mean=torch.sum(torch.sum(candidate_feature_all,dim=2,keepdim=True),dim=3,keepdim=True).expand_as(candidate_feature_all) \
                                /(template_feature.shape[-1]*template_feature.shape[-2])

        representation_c = torch.cat([candidate_feature_mean, candidate_feature_all],1)  #######(N,256*2,6,6,17,17)
        representation_c=representation_c.permute(0,4,5,1,2,3).contiguous().view(representation_c.shape[0]*17*17,representation_c.shape[1],size,size).contiguous()
        weight_c=self.similarity1(representation_c)######(N*17*17,1,6,6)
        weight_c_norm=F.softmax(weight_c.view(weight_c.shape[0],1,-1,1), dim=2).view(representation_c.shape[0]//17//17,17,17,1,size,size).permute(0,3,4,5,1,2).contiguous()
        common_candidate=torch.sum(torch.sum(candidate_feature_all*weight_c_norm,dim=2,keepdim=True),dim=3,keepdim=True)#####(N,1,17,17)
        common_candidata=torch.cat([common_candidate,weight_c_norm.reshape(andidate_feature.shape[0],-1,17,17)],dim=1)
        #representation_c = torch.cat([common_candidate.expand_as(candidate_feature_all),candidate_feature_all],1)
        #representation_c = representation_c.permute(0,4,5,1,2,3).contiguous().view(representation_c.shape[0]*17*17,representation_c.shape[1],size,size).contiguous()
        # return common_template.view(common_template.shape[0],common_template.shape[1],1,1),common_candidate.view(common_candidate.shape[0],common_candidate.shape[1],17,17)
        return {
            'common_template':common_template,
            'common_search':common_candidate,
        }
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

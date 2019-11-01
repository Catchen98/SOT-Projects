# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from IPython import embed
import numpy as np
import os
import cv2
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, pixel_wise_loss, discriminate_loss
from pysot.models.backbone import get_backbone
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from pysot.models.stn import get_stn
from pysot.models.commonsense import get_commonsense_head
from pysot.models.reconstruction import get_decoder
from pysot.models.discriminate import get_discriminator
from pysot.models.head import get_rpn_head, get_refine_head, get_mask_head
from pysot.models.neck import get_neck
from pysot.core.xcorr import xcorr_depthwise

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        # if cfg.DEFORM.DEFORM:
        #     self.kernel_deformconv = DeformConv(in_channels, hidden, kernel_size=kernel_size, padding=1, bias=False)
            # self.search_deformconv = DeformConv(in_channels, hidden, kernel_size=kernel_size, padding=1, bias=False)
        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        # self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
        #                              **cfg.RPN.KWARGS)
        # build commonsenese
        if cfg.COMMONSENSE.COMMONSENSE:
            self.commonsense_head = get_commonsense_head(cfg.COMMONSENSE.TYPE,
                                                         **cfg.COMMONSENSE.KWARGS)
        # build spatial transform network
        if cfg.STN.STN:
            self.stn = get_stn(cfg.STN.TYPE)
        if cfg.RECONSTRUCTION.RECONSTRUCTION:
            self.reconstruction = get_decoder(cfg.RECONSTRUCTION.TYPE,
                                              **cfg.RECONSTRUCTION.KWARGS)
        if cfg.DISCRIMINATOR.DISCRIMINATOR:
            self.discriminate = get_discriminator(cfg.DISCRIMINATOR.TYPE,
                                                  **cfg.DISCRIMINATOR.KWARGS)
        # build mask head
        # if cfg.MASK.MASK:
        #     self.mask_head = get_mask_head(cfg.MASK.TYPE,
        #                                    **cfg.MASK.KWARGS)

        # if cfg.REFINE.REFINE:
        #     self.refine_head = get_refine_head(cfg.REFINE.TYPE)
    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls
    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score
    
    def forward(self, data, idx):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        # mask_gt = data['search_mask'].cuda().permute(0, 3, 1, 2)
        batch_size = template.shape[0]
        # label_cls = data['label_cls'].cuda()
        # label_loc = data['label_loc'].cuda()
        # label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        template = F.interpolate(template,(128,128),mode='bilinear',align_corners=True)
        search = F.interpolate(search,(256,256),mode='bilinear',align_corners=True)
        
        zfs = self.backbone(template)
        xfs = self.backbone(search)
        
        # if cfg.MASK.MASK:
        #     zf = zf[-1]
        #     self.xf_refine = xf[:-1]
        #     xf = xf[-1]
       
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zfs[-2:])
            xf = self.neck(xfs[-2:])
        if cfg.COMMONSENSE.COMMONSENSE:
            zf,xf = self.commonsense_head(zf,xf)
        
        # test = xf
        xcorr = xcorr_depthwise(xf, zf).unsqueeze(1)
        
        # cls, loc = self.rpn_head(zf, xf)
        # cls, loc = self.rpn_head(xcorr)
        
        score=xcorr.reshape(batch_size,-1)
        # scores_threshold=torch.max(score*0.5,dim=1)
        nums = torch.argmax(score,dim=1)
        for i in range(0,len(nums)):
            pos = np.unravel_index(nums[i].detach().cpu().numpy(),(1,xcorr.shape[1],xcorr.shape[2],xcorr.shape[3]))
            
            if cfg.STN.STN:
                replace,theta = self.stn(zf[i,:,:,:].unsqueeze(0),xf[i,:,pos[2]:pos[2]+16,pos[3]:pos[3]+16].unsqueeze(0))
            if i==0:
                thetas=theta
                all_pos=pos
            else:
                thetas=torch.cat([thetas,theta],dim=0)
                all_pos=np.vstack((all_pos,pos))
            xf[i,:,pos[2]:pos[2]+16,pos[3]:pos[3]+16]=replace
            # xf[i,:,pos[2]:pos[2]+16,pos[3]:pos[3]+16]=zf[i,:,:,:].unsqueeze(0)
            # if i==0:
            #     mask=torch.where(score[i]>scores_threshold[0][i],torch.full_like(score[i],1),torch.full_like(score[i],0)).reshape(1,1,xcorr.shape[2],xcorr.shape[3])
            # else:
            #     mask=torch.cat([mask,torch.where(score[i]>scores_threshold[0][i],torch.full_like(score[i],1),
            #                                      torch.full_like(score[i],0)).reshape(1,1,xcorr.shape[2],xcorr.shape[3])],dim=0)
        # mask=mask.detach().cpu().numpy()
    
        # mask=T.Resize((batch_size,mask.shape[1],search.shape[2],search.shape[3]))(mask)
        
        if cfg.RECONSTRUCTION.RECONSTRUCTION:
            pre_search = self.reconstruction(xf,thetas,all_pos,zfs)
        # embed()
        heatmap = F.interpolate(xcorr[0,:,:,:].unsqueeze(0),(256,256),mode='bilinear',align_corners=True).view(256,256)
        heatmap = heatmap.data.cpu().numpy()
        heatmap/= np.max(heatmap)
        heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET))
        heatmap = heatmap/255+pre_search[0,:,:,:].data.cpu().numpy().transpose(1,2,0)
        # heatmap = heatmap/255+(search[0,:,:,:]/255.0).data.cpu().numpy().transpose(1,2,0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap*255
        # embed()
        # exit()
        if cfg.DISCRIMINATOR.DISCRIMINATOR:
            realscore = self.discriminate(search/255.0)
            reconsscore = self.discriminate(pre_search)
        # image_mask=search.detach().cpu().numpy()*mask
        # get loss
        reconstruction_loss = pixel_wise_loss(pre_search, search / 255.0)
        # tem_loss = pixel_wise_loss(test_template, template/255.0)
        device = torch.device("cuda")
        real_label = torch.full((batch_size,), 1, device=device)
        fake_label = torch.full((batch_size,), 0, device=device)
        r_d_loss = discriminate_loss(realscore, real_label)
        f_d_loss = discriminate_loss(reconsscore, fake_label)
        f_g_loss = discriminate_loss(reconsscore, real_label)
        # cls = self.log_softmax(cls)
        # cls_loss = select_cross_entropy_loss(cls, label_cls)
        # loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        D_loss = r_d_loss + f_d_loss #+ sr_d_loss + ksf_d_loss
        G_loss = f_g_loss + reconstruction_loss #+ smask_loss#+ ksf_g_loss
        E_loss = reconstruction_loss
        save_path = './test_results/mulimage/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if idx % 200 == 0:
        #     cv2.imwrite(os.path.join(save_path, 'mask_{}.jpg'.format(idx)), save_mask)
            save_template=template[0,:,:,:].detach().cpu().numpy().transpose(1,2,0)
            save_search = search[0,:,:,:].detach().cpu().numpy().transpose(1,2,0)
            save_pre=(pre_search[0,:,:,:]*255).detach().cpu().numpy().transpose(1,2,0)
            cv2.imwrite(os.path.join(save_path,'template_{}.jpg'.format(idx)),np.uint8(save_template))
            cv2.imwrite(os.path.join(save_path,'pre_search_{}.jpg'.format(idx)),np.uint8(save_pre))
            cv2.imwrite(os.path.join(save_path,'heatmap_{}.jpg'.format(idx)),np.uint8(heatmap))
            cv2.imwrite(os.path.join(save_path,'search_{}.jpg'.format(idx)),np.uint8(save_search))
        outputs = {}
        outputs['E_loss'] = E_loss
        outputs['G_loss'] = G_loss
        outputs['D_loss'] = D_loss
        # outputs['pre_search'] = (pre_search[0,:,:,:]*255).detach().cpu().numpy()
        outputs['template_gt'] = template[0,:,:,:].detach().cpu().numpy()
        outputs['test_search']= (pre_search[0,:,:,:]*255).detach().cpu().numpy()
        outputs['heatmap'] = heatmap.transpose(2,0,1)#
        outputs['search_gt'] = search[0,:,:,:].detach().cpu().numpy()
        # outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
        #     cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.MASK_WEIGHT
        # outputs['cls_loss'] = cls_loss
        # outputs['loc_loss'] = loc_loss

        # if cfg.MASK.MASK:
        #     # TODO
        #     mask, self.mask_corr_feature = self.mask_head(zf, xf)
        #     mask_loss = None
        #     outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
        #     outputs['mask_loss'] = mask_loss
        return outputs

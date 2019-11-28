# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import os
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.manytools import *


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)
     
        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf
        self.z=z
    def track(self, x, idx, model_name,video_name):
        xf = self.backbone(x)
        batch_size=xf.shape[0]
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        if cfg.DEFORMABLE.DEFORMABLE or cfg.COMMONSENSE.COMMONSENSE:
            cls_outputs, loc_outputs = self.rpn_head(self.zf, xf)
            xcorr=cls_outputs['xcorr']
            index=compute_argmaxxcorr(xcorr)
            if cfg.COMMONSENSE.COMMONSENSE:
                kernel_matrix=cls_outputs['kernel_matrix']
                search_matrix=cls_outputs['search_matrix'][:,:,:,:,index[0],index[1]]
                kernel_matrix=draw_heatmap(kernel_matrix,self.z.squeeze())
                search_matrix=draw_heatmap(search_matrix.reshape(1,1,search_matrix.shape[2],search_matrix.shape[3])
                                ,x[:,:,index[0]*8:index[0]*8+127,index[1]*8:index[1]*8+127].squeeze())
                vis_scf=torch.mean(cls_outputs['scf'][:,:,:,:,index[0],index[1]],dim=1,keepdim=True)
            else:
                vis_scf=torch.mean(cls_outputs['scf'][:,:,index[0]:index[0]+self.zf.shape[2],index[1]:index[1]+self.zf.shape[3]],dim=1,keepdim=True)
            vis_ksf=torch.mean(cls_outputs['ksf'],dim=1,keepdim=True)
            vis_kcf=torch.mean(cls_outputs['kcf'],dim=1,keepdim=True)
            vis_ssf=torch.mean(cls_outputs['ssf'],dim=1,keepdim=True)
            vis_ksf=draw_heatmap(vis_ksf,self.z.squeeze())
            vis_kcf=draw_heatmap(vis_kcf,self.z.squeeze())
            vis_ssf=draw_heatmap(vis_ssf,x[:,:,index[0]*8:index[0]*8+127,index[1]*8:index[1]*8+127].squeeze())
            vis_scf=draw_heatmap(vis_scf,x[:,:,index[0]*8:index[0]*8+127,index[1]*8:index[1]*8+127].squeeze())
            if video_name:
                savefeature_root='./results/save_features/'+model_name+'/'+video_name+'/'
                if not os.path.exists(savefeature_root):
                    os.makedirs(savefeature_root+'template_sfeatures/')
                    os.makedirs(savefeature_root+'template_cfeatures/')
                    os.makedirs(savefeature_root+'search_sfeatures/')
                    os.makedirs(savefeature_root+'search_cfeatures/')
                save_features(savefeature_root+'template_cfeatures/template_cf.pth',cls_outputs['kcf'])
                if cfg.COMMONSENSE.COMMONSENSE:
                    search_sf=cls_outputs['ssf']
                    search_cf=cls_outputs['scf'][:,:,:,:,index[0],index[1]]
                    kernel_sf=cls_outputs['ksf']
                    kernel_cf=cls_outputs['kcf']
                    if kernel_sf.shape[1]<kernel_cf.shape[1]:
                        save_features(savefeature_root+'template_sfeatures/template_sf.pth',torch.cat([kernel_sf,kernel_sf[:,-1,:,:].unsqueeze(1)],dim=1))
                    else:
                        save_features(savefeature_root+'template_sfeatures/template_sf.pth',kernel_sf)
                    save_features(savefeature_root+'search_sfeatures/search_sf_{}.pth'.format(idx),torch.cat([search_sf,search_sf[:,-1,:,:].unsqueeze(1)],dim=1),index)
                    save_features(savefeature_root+'search_cfeatures/search_cf_{}.pth'.format(idx),search_cf)
                else:
                    search_sf=cls_outputs['ssf']
                    search_cf=cls_outputs['scf']
                    save_features(savefeature_root+'template_sfeatures/template_sf.pth',cls_outputs['ksf'])
                    save_features(savefeature_root+'search_sfeatures/search_sf_{}.pth',search_sf,index)
                    save_features(savefeature_root+'search_cfeatures/search_cf_{}.pth',search_cf,index)
                
            return{
                   'cls':cls_outputs['xcorr'],
                   'loc':loc_outputs['xcorr'],
                   'original_xcorr':cls_outputs['original_xcorr'],
                   'kernel_matrix': kernel_matrix if cfg.COMMONSENSE.COMMONSENSE else None,
                   'search_matrix': search_matrix if cfg.COMMONSENSE.COMMONSENSE else None,
                   'vis_ksf': vis_ksf,
                   'vis_kcf': vis_kcf,
                   'vis_ssf': vis_ssf,
                   'vis_scf': vis_scf,
                 }
        else:
            cls_outputs, loc_outputs = self.rpn_head(self.zf, xf)
            cls=cls_outputs['xcorr']
            loc=loc_outputs['xcorr']
            
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

    def forward(self, data,idx):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        if cfg.DEFORMABLE.DEFORMABLE:
            cls_outputs,loc_outputs=self.rpn_head(zf,xf)
            cls=cls_outputs['xcorr']
            loc=loc_outputs['xcorr']
            xcorr=cls
            ksf=cls_outputs['ksf']
            kcf=cls_outputs['kcf']
        elif cfg.COMMONSENSE.COMMONSENSE:
            cls_outputs,loc_outputs=self.rpn_head(zf,xf)
            cls=cls_outputs['xcorr']
            loc=loc_outputs['xcorr']
            xcorr=cls
            ksf=cls_outputs['ksf']
            kcf=cls_outputs['kcf']
        else:
            cls_outputs, loc_outputs = self.rpn_head(zf, xf)
            cls=cls_outputs['xcorr']
            loc=loc_outputs['xcorr']
            xcorr=cls
        # embed()
        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['template_gt']=template
        outputs['search_gt']=search
        outputs['cls']=xcorr
        if cfg.DEFORMABLE.DEFORMABLE:
            outputs['vis_sf']=ksf
            outputs['vis_cf']=kcf
        if cfg.COMMONSENSE.COMMONSENSE:
            outputs['vis_sf']=ksf
            outputs['vis_cf']=kcf
        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs

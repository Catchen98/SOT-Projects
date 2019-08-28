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
import cv2
import os
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, pixel_wise_loss, discriminate_loss
from pysot.models.backbone import get_backbone
from pysot.models.disentangle import get_disentangle
from pysot.models.reconstruction import get_decoder
from pysot.models.discriminate import get_discriminator
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        # build disentangle
        if cfg.DISENTANGLE.DISENTANGLE:
            self.split = get_disentangle(cfg.DISENTANGLE.TYPE,
                                         **cfg.DISENTANGLE.KWARGS)
        # build reconstruction
        if cfg.RECONSTRUCTION.RECONSTRUCTION:
            self.reconstruct_mask = get_decoder(cfg.RECONSTRUCTION.TYPE,
                                                **cfg.RECONSTRUCTION.MKWARGS)
            self.reconstruct_image = get_decoder(cfg.RECONSTRUCTION.TYPE,
                                                 **cfg.RECONSTRUCTION.IKWARGS)
        if cfg.DISCRIMINATOR.DISCRIMINATOR:
            self.discriminate = get_discriminator(cfg.DISCRIMINATOR.TYPE,
                                                  **cfg.DISCRIMINATOR.KWARGS)
        #self.unfold = nn.Unfold(kernel_size=(127, 127), stride=32)
    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)##extract feature

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
                # 'tem_feature':kernel,
                # 'cls_feature': cls_feature,
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
        #search = data['search'].cuda()
        template_mask = data['template_mask'].cuda().permute(0,3,1,2)
        #template_box = data['template_box']
        #search = self.unfold(search).view(-1, search.shape[1], 127, 127).contiguous()
        batch_size = template.size(0)
        #repeat_size = search.size(0)//batch_size

        # get feature
        zf = self.backbone(template)
        #xf = self.backbone(search)

        if cfg.DISENTANGLE.DISENTANGLE:
            #results = self.split(zf, xf, template_box)
            results = self.split(zf)
        if cfg.RECONSTRUCTION.RECONSTRUCTION:
            k_mask_img = self.reconstruct_mask(results['k_mask'])
            k_image = self.reconstruct_image(results['restruct_kimg'])
            #k_image = F.sigmoid(k_image)
            #ks_image = self.reconstruct_image(results['restruct_ksimg'])
        if cfg.DISCRIMINATOR.DISCRIMINATOR:
            k_realscore = self.discriminate(template/255)
            #s_realscore = self.discriminate(search/255)
            k_reconsscore = self.discriminate(k_image)
            #ks_reconsscore = self.discriminate(ks_image)

        # -------------mask_loss
        #k_mask_img = F.sigmoid(k_mask_img)
        k_mask_loss = k_mask_img.view(k_mask_img.size(0),-1)
        mask_gt=(template_mask//255)[:,0,:,:].view(template_mask.size(0),-1)
        mask_gt = mask_gt.type(torch.float)
        mask_loss = nn.BCELoss()(k_mask_loss,mask_gt)
        # -------------reconstruct_loss

        k_reconsimg_loss = pixel_wise_loss(k_image, template / 255.0)
        #ks_reconsimg_loss = pixel_wise_loss(ks_image, search / 255)
        restruct_loss = k_reconsimg_loss #+ ks_reconsimg_loss
        # ------------discriminator_loss
        device = torch.device("cuda")
        real_label = torch.full((batch_size,), 1, device=device)
        fake_label = torch.full((batch_size,), 0, device=device)
        kr_d_loss = discriminate_loss(k_realscore, real_label)
        kf_d_loss = discriminate_loss(k_reconsscore, fake_label)
        kf_g_loss = discriminate_loss(k_reconsscore, real_label)
        # sr_d_loss = discriminate_loss(s_realscore, real_label.repeat(repeat_size,1))
        # sf_d_loss = discriminate_loss(ks_reconsscore, fake_label.repeat(repeat_size,1))
        D_loss = kr_d_loss + kf_d_loss #+ sr_d_loss + sf_d_loss
        G_loss = kf_g_loss + restruct_loss + mask_loss
        E_loss = kf_g_loss + mask_loss + restruct_loss
        outputs = {}
        outputs['E_loss'] = E_loss
        outputs['G_loss'] = G_loss
        outputs['D_loss'] = D_loss
        outputs['mask_loss'] = mask_loss
        outputs['k_reconsimg_loss'] = k_reconsimg_loss
        #outputs['ks_reconsimg_loss'] = ks_reconsimg_loss
        outputs['kf_g_loss'] = kf_g_loss
        outputs['kr_d_loss'] = kr_d_loss
        outputs['kf_d_loss'] = kf_d_loss
        # outputs['sr_d_loss'] = sr_d_loss
        # outputs['sf_d_loss'] = sf_d_loss

        save_k_mask = k_mask_img[0, 0, :, :].detach().cpu().numpy()
        mean = np.mean(save_k_mask)
        save_k_mask = np.int64(save_k_mask>mean)
        save_k_mask = (save_k_mask * 255)

        save_mask_gt = (data['template_mask'][0, :, :, :]).detach().cpu().numpy()
        save_mask_gt = np.uint8(save_mask_gt)

        save_template = (data['template'][0, :, :, :]).detach().cpu().numpy()
        save_template_mask = save_template + save_k_mask
        save_template_mask = save_template_mask/np.max(save_template_mask)
        save_template_mask = (save_template_mask*255)
        save_reconstruct_t = (k_image[0,:,:,:]*255).detach().cpu().numpy()
        #save_reconstruct_s = (ks_image[12,:,:,:]*255).detach().cpu().numpy()
        save_template_mask = np.uint8(save_template_mask.transpose(1, 2, 0))
        save_template = np.uint8(save_template.transpose(1, 2, 0))
        save_k_mask = np.uint8(save_k_mask)
        save_reconstruct_t = np.uint8(save_reconstruct_t.transpose(1,2,0))
        #save_reconstruct_s = np.uint8(save_reconstruct_s.transpose(1,2,0))

        saveimage_path = './simpleG_img/'
        if not os.path.exists(saveimage_path):
            os.makedirs(saveimage_path)
        if idx % 200 == 0:
            cv2.imwrite(saveimage_path + '/template_{}.jpg'.format(idx), save_template)
            cv2.imwrite(saveimage_path + '/k_mask_{}.jpg'.format(idx), save_k_mask)
            cv2.imwrite(saveimage_path + '/mask_gt_{}.jpg'.format(idx), save_mask_gt)
            cv2.imwrite(saveimage_path + 'template_mask_{}.jpg'.format(idx), save_template_mask)
            cv2.imwrite(saveimage_path + '/reconstruct_template_{}.jpg'.format(idx),save_reconstruct_t)
            #cv2.imwrite(saveimage_path + '/reconstruct_search_{}.jpg'.format(idx),save_reconstruct_s)

        return outputs

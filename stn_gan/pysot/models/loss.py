# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F
import torch.nn as nn

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0:
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


# def select_cross_entropy_loss(pred, label):
#     pred = pred.view(-1, 2)
#     label = label.view(-1)
#     pos = label.data.eq(1).nonzero().squeeze().cuda()
#     neg = label.data.eq(0).nonzero().squeeze().cuda()
#     loss_pos = get_cls_loss(pred, label, pos)
#     loss_neg = get_cls_loss(pred, label, neg)
#     return loss_pos * 0.5 + loss_neg * 0.5
def select_cross_entropy_loss(pred, label):
    label = label.to(torch.long)
    return F.cross_entropy(pred, label)

def pixel_wise_loss(restruct_img,original_img):
    loss_function = nn.L1Loss()
    loss = loss_function(restruct_img,original_img)
    return loss
def discriminate_loss(score,label):
    loss_function = nn.BCELoss()
    loss = loss_function(score,label)
    return loss
def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)

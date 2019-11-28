# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import sys
import time
import math
import json
import random
import numpy as np
from visdom import Visdom
from tqdm import tqdm
import torch
import torch.nn as nn
from IPython import embed
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import cv2
from torch.utils.data.distributed import DistributedSampler
sys.path.append(os.getcwd())
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
# from pysot.datasets.FT3Ddataset import TrkDataset
from pysot.core.config import cfg
from pysot.models.manytools import *
from pysot.models.misc import flow_to_color


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--model_name', type=str, default='',
                    help='name of model')
parser.add_argument('--snapshot',type=str,default='',
                    help='pretrain model')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
parser.add_argument('--vis', action='store_true',
                    help='whether visualzie result')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    # if get_world_size() > 1:
    #     train_sampler = DistributedSampler(train_dataset)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=cfg.TRAIN.BATCH_SIZE ,
    #                           num_workers=cfg.TRAIN.NUM_WORKERS,
    #                           pin_memory=True,
    #                           sampler=train_sampler)
    train_loader = DataLoader(train_dataset,batch_size=cfg.TRAIN.BATCH_SIZE*torch.cuda.device_count(),
                              num_workers=cfg.TRAIN.NUM_WORKERS*torch.cuda.device_count(),shuffle=True,
                              pin_memory=True,drop_last=True)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    if args.model_name=='rpn':
        trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                            model.backbone.parameters()),
                            'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]
        
        if cfg.ADJUST.ADJUST:
            trainable_params += [{'params': model.neck.parameters(),
                            'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.rpn_head.parameters(),
                            'lr': cfg.TRAIN.BASE_LR}]
    
    
    if cfg.COMMONSENSE.COMMONSENSE:
        trainable_params += [{'params': model.rpn_head.cls.commonsense.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.rpn_head.loc.commonsense.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.rpn_head.cls.commonhead.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.rpn_head.loc.commonhead.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
    if cfg.TSF.TSF:
        trainable_params += [{'params': model.rpn_head.cls.tsf.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.rpn_head.loc.tsf.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
    if cfg.DEFORMABLE.DEFORMABLE:
        trainable_params += [{'params': model.rpn_head.cls.deform_layer.parameters(),
                            'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.rpn_head.loc.deform_layer.parameters(),
                            'lr': cfg.TRAIN.BASE_LR}]
    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.LR.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)


def train(train_loader, model, optimizer, lr_scheduler, model_name):
    if args.vis:
        vis = Visdom(env=args.model_name)
        clsloss_win = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1)).cpu(),
                            opts=dict(xlabel='image_number', ylabel='cls_loss', title='cls_loss',
                                    legend=['cls_Loss']))
        locloss_win = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1)).cpu(),
                            opts=dict(xlabel='image_number', ylabel='loc_loss', title='loc_loss',
                                    legend=['loc_Loss']))
        search_win = vis.image(np.random.rand(3, 255, 255),
                            opts=dict(title='search'))
        template_win = vis.image(np.random.rand(3, 127, 127),
                            opts=dict(title='template'))
        heatmap_win = vis.image(np.random.rand(3, 255, 255),
                            opts=dict(title='heatmap'))
        if cfg.COMMONSENSE.COMMONSENSE or cfg.DEFORMABLE.DEFORMABLE:
            feature_win = vis.image(np.random.rand(3, 127, 127),
                                opts=dict(title='Sfeature'))
            commonsense_win = vis.image(np.random.rand(3,127,127),
                                opts=dict(title='Cfeature'))
        if cfg.TSF.TSF:
            tsfloss_win = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1)).cpu(),
                            opts=dict(xlabel='image_number', ylabel='tsf_loss', title='tsf_loss',
                                    legend=['tsf_Loss']))
            # warped_heatmap_win = vis.image(np.random.rand(3, 255, 255),
            #                     opts=dict(title='warped_heatmap'))
            ztsf_win = vis.image(np.random.rand(3, 127, 127),
                                opts=dict(title='vis_ztsf'))
            xtsf_win = vis.image(np.random.rand(3, 127, 127),
                                opts=dict(title='vis_xtsf'))
            zf_win = vis.image(np.random.rand(3, 127, 127),
                                opts=dict(title='vis_zf'))
            xf_win = vis.image(np.random.rand(3, 127, 127),
                                opts=dict(title='vis_xf'))
            matrix1_win = vis.image(np.random.rand(3, 127, 127),
                                opts=dict(title='vis_matrix1'))
            matrix2_win = vis.image(np.random.rand(3, 127, 127),
                                opts=dict(title='vis_matrix2'))
    
    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    print('dataset length:', len(train_loader.dataset))
    num_per_epoch = len(train_loader.dataset) // \
                    cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch
    savemodel_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, model_name,)

    if not os.path.exists(savemodel_path):
        os.makedirs(savemodel_path)

    print(num_per_epoch)

    for idx, data in enumerate(tqdm(train_loader)):
        start_time = time.time()
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch
         
            if epoch % 1 == 0:
                torch.save(
                    {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()
                     },
                    savemodel_path + '/checkpoint_e%d.pth' % (epoch))
            
            if epoch == cfg.TRAIN.EPOCH:
                return
            # lr_scheduler.step(epoch)
            # cur_lr = lr_scheduler.get_cur_lr()
            lr_scheduler.step(epoch)
        torch.save(
                    {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()
                     },
                    savemodel_path + '/checkpoint_e%d.pth' % (epoch))
        tb_idx = idx
        # train model
        outputs = model(data,idx)
        loss = outputs['total_loss'].mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # draw heatmap
        heatmap=outputs['cls'][0,:,:,:].reshape(1,2,-1).permute(0,2,1)
        heatmap=F.softmax(heatmap,dim=2)[0,:,1].view(1,5,outputs['cls'].shape[2],outputs['cls'].shape[3])
        heatmap,indicesn=torch.max(heatmap,dim=1,keepdim=True)
        heatmap=draw_heatmap(heatmap,outputs['search_gt'][0,:,:,:])
        if idx % 50 == 0:
            print(args.model_name,'time', time.time() - start_time, 'epoch: ', epoch, 'sample: ', idx, 'loss: ', loss.item(),
                   'cls_loss:', outputs['cls_loss'].mean().item(), 'loc_loss:', outputs['loc_loss'].mean().item()
                   )
            if cfg.TSF.TSF:
                print(args.model_name,'tsf_loss:',outputs['tsf_loss'].mean().item())
        vis.line(Y=np.array([outputs['cls_loss'].mean().item()]), X=np.array([idx]), win=clsloss_win, update='append')
        vis.line(Y=np.array([outputs['loc_loss'].mean().item()]), X=np.array([idx]), win=locloss_win, update='append')
        vis.image(heatmap.transpose(2,0,1),win=heatmap_win,opts=dict(title='heatmap'))
        vis.image(np.uint8(outputs['search_gt'][0,:,:,:].data.cpu().numpy()),win=search_win,opts=dict(title='search'))
        vis.image(np.uint8(outputs['template_gt'][0,:,:,:].data.cpu().numpy()),win=template_win,opts=dict(title='template'))
        if cfg.COMMONSENSE.COMMONSENSE or cfg.DEFORMABLE.DEFORMABLE:
            # vis feature
            sf=outputs['vis_sf']
            cf=outputs['vis_cf']
            sf=torch.mean(sf[0,:,:,:],dim=0,keepdim=True).unsqueeze(0)
            sf=draw_heatmap(sf,outputs['template_gt'][0,:,:,:])
            cf=torch.mean(cf[0,:,:,:],dim=0,keepdim=True).unsqueeze(0)
            cf=draw_heatmap(cf,outputs['template_gt'][0,:,:,:])
            vis.image(sf.transpose(2,0,1),win=feature_win,opts=dict(title='Sfeature'))
            vis.image(cf.transpose(2,0,1),win=commonsense_win,opts=dict(title='Cfeature'))
        if cfg.TSF.TSF:
            zx_grid=outputs['vis_zxgrid']
            xz_grid=outputs['vis_xzgrid']
            vis_zxgrid=torch.from_numpy(flow_to_color(zx_grid[0,:,:,:].squeeze().data.cpu().numpy())).float()
            vis_zxgrid = F.interpolate(vis_zxgrid.permute(2,0,1).unsqueeze(0),(127,127),mode='bilinear',align_corners=True)
            vis_xzgrid=torch.from_numpy(flow_to_color(xz_grid[0,:,:,:].squeeze().data.cpu().numpy())).float()
            vis_xzgrid = F.interpolate(vis_xzgrid.permute(2,0,1).unsqueeze(0),(127,127),mode='bilinear',align_corners=True)
            # warped_heatmap=outputs['warped_xcorr'][0,:,:,:].reshape(1,2,-1).permute(0,2,1)
            # warped_heatmap=F.softmax(warped_heatmap,dim=2)[0,:,1].view(1,5,17,17)
            # warped_heatmap,indicesn=torch.max(warped_heatmap,dim=1,keepdim=True)
            # warped_heatmap=draw_heatmap(warped_heatmap,outputs['search_gt'][0,:,:,:])
            vis_ztsf = (outputs['vis_ztsf']).squeeze().data.cpu().numpy()
            vis_xtsf = (outputs['vis_xtsf']).squeeze().data.cpu().numpy()
            vis_zf = (outputs['vis_zf']).squeeze().data.cpu().numpy()
            vis_xf = (outputs['vis_xf']).squeeze().data.cpu().numpy()
            # vis.image(warped_heatmap.transpose(2,0,1),win=warped_heatmap_win,opts=dict(title='warped_heatmap'))
            vis.image(np.uint8(vis_zxgrid.squeeze().data.cpu().numpy()),win=matrix1_win,opts=dict(title='vis_zxgrid'))
            vis.image(np.uint8(vis_xzgrid.squeeze().data.cpu().numpy()),win=matrix2_win,opts=dict(title='vis_xzgrid'))
            vis.image(np.uint8(vis_ztsf),win=ztsf_win,opts=dict(title='vis_ztsf'))
            vis.image(np.uint8(vis_xtsf),win=xtsf_win,opts=dict(title='vis_xtsf'))
            vis.image(np.uint8(vis_zf),win=zf_win,opts=dict(title='vis_zf'))
            vis.image(np.uint8(vis_xf),win=xf_win,opts=dict(title='vis_xf'))
            vis.line(Y=np.array([outputs['tsf_loss'].mean().item()]), X=np.array([idx]), win=tsfloss_win, update='append')
def main():
    
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)

    # create model
    model = ModelBuilder().cuda().train()
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)
    if args.snapshot:
        print('load pretrain siamrpn')
        # model = load_pretrain(model,'modelzoo/siamrpn_r50_l234_dwxcorr_otb/model.pth')
        model = load_pretrain(model, args.snapshot)
    # dist_model = DistModule(model)
    dist_model = model.cuda().train()
    dist_model = torch.nn.DataParallel(dist_model).train()
    # load pretrained backbone weights
    if cfg.TRAIN.RESUME:
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        dist_model, optimizer, cfg.TRAIN.START_EPOCH = restore_from(dist_model, optimizer, cfg.TRAIN.RESUME)

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(dist_model.module,
                                           cfg.TRAIN.START_EPOCH)

    # resume training
    # if cfg.TRAIN.RESUME:
    #     logger.info("resume from {}".format(cfg.TRAIN.RESUME))
    #     assert os.path.isfile(cfg.TRAIN.RESUME), \
    #         '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
    #     model, optimizer, cfg.TRAIN.START_EPOCH = \
    #         restore_from(model, optimizer, cfg.TRAIN.RESUME)
    #     dist_model = DistModule(model)

    # logger.info(lr_scheduler)
    # logger.info("model prepare done")
    torch.autograd.set_detect_anomaly(True)
    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, args.model_name)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()

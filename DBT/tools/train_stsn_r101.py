from __future__ import division
import argparse
import os
import tqdm
import random
from visdom import Visdom
import torch
import numpy as np
from IPython import embed
from collections import OrderedDict
import mmcv
from mmcv import Config
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import build_dataset, DATASETS, build_dataloader
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--vis', action='store_true',
                    help='whether visualzie result')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, 'module'):
        model = model.module
    
    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)
        
        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu

def main():
    args = parse_args()
    if args.vis:
        vis = Visdom(env=args.model_name)
        clsloss_win = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1)).cpu(),
                            opts=dict(xlabel='image_number', ylabel='cls_loss', title='cls_loss',
                                    legend=['cls_Loss']))
        locloss_win = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1)).cpu(),
                            opts=dict(xlabel='image_number', ylabel='loc_loss', title='loc_loss',
                                    legend=['loc_Loss']))
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    # print('train')
    # embed()
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if cfg.load_from:
        checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = datasets[0].CLASSES
    
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    
    data_loader = build_dataloader(
        datasets[0],
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    model.train()
    
    optimizer = build_optimizer(model, cfg.optimizer)
    
    check_video=None
    num_per_epoch=len(data_loader)//cfg.total_epochs
    start_epoch=0
    meta=None
    epoch=start_epoch
    for e in range(cfg.total_epochs):
        for i, data in enumerate(data_loader):
            # if epoch != i // num_per_epoch + start_epoch:
            #     epoch = i // num_per_epoch + start_epoch
            
            if len(data['gt_bboxes'].data[0][0]) == 0:
                continue
            reference_id=(data['img_meta'].data[0][0]['filename'].split('/')[-1]).split('.')[0]
            video_id=data['img_meta'].data[0][0]['filename'].split('/')[-2]
            before=max(i-13,i-int(reference_id))
            # after=min(i)
            reference=data['img'].data[0]
            if epoch>5:
                j=random.randint(before,i)
                support=(datasets[0][j]['img'].data).unsqueeze(0)
                support_id=(datasets[0][j]['img_meta'].data['filename'].split('/')[-1]).split('.')[0]
                svideo_id=(datasets[0][j]['img_meta'].data['filename'].split('/')[-2])
            else:
                support=reference
                support_id=reference_id
                svideo_id=video_id
            
            # data['img']=torch.cat([support,reference],dim=0)
            
            losses=model(return_loss=True, **data)
            
            loss, log_vars = parse_losses(losses)
            if np.isnan(loss.item()):
                embed()
                exit()
            
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            # if np.isnan(loss.item()):
            #     loss.backward(retain_graph=False)
            #     optimizer.zero_grad()
            #     continue
            # optimizer.zero_grad()
            # loss.backward(retain_graph=False)
            # optimizer.step()
            
            if epoch % 1 == 0:
                if meta is None:
                    meta = dict(epoch=epoch + 1, iter=i)
                else:
                    meta.update(epoch=epoch + 1, iter=i)
                checkpoint = {
                    'meta': meta,
                    'state_dict': weights_to_cpu(model.state_dict())
                }
                if optimizer is not None:
                    checkpoint['optimizer'] = optimizer.state_dict()
                mmcv.mkdir_or_exist(os.path.dirname(args.work_dir))
                filename=os.path.join(args.work_dir,'epoch_{}.pth'.format(epoch))
                torch.save(checkpoint,filename)
            
            print(args.work_dir.split('/')[-2],'i:',i,'epoch:',epoch,'video_id:',video_id,'support_id:',support_id,'reference_id:',reference_id,'loss_rpn_cls:',log_vars['loss_rpn_cls'],'loss_rpn_bbox:',log_vars['loss_rpn_bbox'],
                    'loss_cls:',log_vars['loss_cls'],'acc:',log_vars['acc'],'loss_bbox:',log_vars['loss_bbox'])
            
        epoch+=1
            
        
    # train_detector(
    #     model,
    #     datasets,
    #     cfg,
    #     distributed=distributed,
    #     validate=args.validate,
    #     logger=logger)


if __name__ == '__main__':
    main()

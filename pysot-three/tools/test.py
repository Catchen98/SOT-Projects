# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import cv2
import torch
import numpy as np
from IPython import embed
from visdom import Visdom
sys.path.append(os.getcwd())
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from pysot.models.manytools import *


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--model_name',default='', type=str,
        help='name of result')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

def main(checkpoint):
    # load config
    cfg.merge_from_file(args.config)
    if args.vis:
        vis=Visdom(env='baseline')
        image_win = vis.image(np.random.rand(3, 500, 500),opts=dict(title='image'))
        heatmap_win = vis.image(np.random.rand(3,500,500),opts=dict(title='heatmap'))
        if cfg.COMMONSENSE.COMMONSENSE or cfg.DEFORMABLE.DEFORMABLE:
            kernelmatrix_win = vis.image(np.random.rand(3,200,200),opts=dict(title='kernel_matrix'))
            searchmatrix_win = vis.image(np.random.rand(3,200,200),opts=dict(title='search_matrix'))
            kfeature_win = vis.image(np.random.rand(3, 127, 127),
                                opts=dict(title='KSfeature'))
            kcommonsense_win = vis.image(np.random.rand(3,127,127),
                                opts=dict(title='KCfeature'))
            sfeature_win = vis.image(np.random.rand(3, 255, 255),
                                opts=dict(title='SSfeature'))
            scommonsense_win = vis.image(np.random.rand(3,255,255),
                                opts=dict(title='SCfeature'))
            original_heatmap_win = vis.image(np.random.rand(3,500,500),opts=dict(title='original_heatmap'))
    

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    dataset_root = os.path.join('/databack/data',args.dataset)
    # create model
    model = ModelBuilder()

    # load model
    # model = load_pretrain(model, args.snapshot).cuda().eval()
    model=load_pretrain(model,checkpoint).cuda().eval()

    # build tracker
    tracker = build_tracker(model)
   
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # model_name = args.snapshot.split('/')[-1].split('.')[0]
    model_name = args.model_name
    total_lost = 0
    save_image_path='./results/save_image/'+args.model_name+'/'+args.video
    if not os.path.exists(save_image_path+'/images/'):
        os.makedirs(save_image_path+'/images/')
    if not os.path.exists(save_image_path+'/heatmaps/'):
        os.makedirs(save_image_path+'/heatmaps/')
    if not os.path.exists(save_image_path+'/original_heatmaps/'):
        os.makedirs(save_image_path+'/original_heatmaps/')
    if not os.path.exists(save_image_path+'/matrixs/'):
        os.makedirs(save_image_path+'/matrixs/')
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img,idx,args.model_name,args.video)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if args.video:
                        pred_bboxes.append(pred_bbox)
                    else:
                        if overlap > 0:
                            # not lost
                            pred_bboxes.append(pred_bbox)
                        else:
                            # lost object
                            pred_bboxes.append(2)
                            frame_counter = idx + 5 # skip 5 frames
                            lost_number += 1
                    
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imwrite(save_image_path+'/images/image_{}.jpg'.format(idx),img)
                
                    heatmap=outputs['heatmap']
                    cv2.putText(heatmap, str(idx), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    cv2.imwrite(save_image_path+'/heatmaps/heatmap_{}.jpg'.format(idx),heatmap)
                    
                    b,g,r=cv2.split(heatmap)
                    heatmap=cv2.merge([r,g,b])
                    b,g,r=cv2.split(img)
                    img=cv2.merge([r,g,b])
                    vis.image(heatmap.transpose(2,0,1),win=heatmap_win,opts=dict(title='heatmap'))
                    vis.image(img.transpose(2,0,1),win=image_win,opts=dict(title='image'))
                    if cfg.COMMONSENSE.COMMONSENSE or cfg.DEFORMABLE.DEFORMABLE:
                        kernel_matrix=outputs['kernel_matrix']
                        search_matrix=outputs['search_matrix']
                        vis.image(kernel_matrix.transpose(2,0,1),win=kernelmatrix_win,opts=dict(title='kernel_matrix'))
                        vis.image(search_matrix.transpose(2,0,1),win=searchmatrix_win,opts=dict(title='search_matrix'))
                        r,g,b=cv2.split(search_matrix)
                        search_matrix=cv2.merge([b,g,r])
                        cv2.imwrite(save_image_path+'/matrixs/searchmatrix_{}.jpg'.format(idx),search_matrix)

                    # cv2.imshow(video.name, img)
                    # cv2.waitKey(1)
                    
                        
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:  ####OTB
                    outputs = tracker.track(img,idx,args.model_name,args.video)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                # embed()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # embed()
                    heatmap=outputs['heatmap']
                    cv2.putText(heatmap, str(idx), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    cv2.imwrite(save_image_path+'/images/image_{}.jpg'.format(idx),img)
                    cv2.imwrite(save_image_path+'/heatmaps/heatmap_{}.jpg'.format(idx),heatmap)
                    b,g,r=cv2.split(heatmap)
                    heatmap=cv2.merge([r,g,b])
                    b,g,r=cv2.split(img)
                    img=cv2.merge([r,g,b])
                    vis.image(heatmap.transpose(2,0,1),win=heatmap_win,opts=dict(title='heatmap'))
                    vis.image(img.transpose(2,0,1),win=image_win,opts=dict(title='image'))
                    # cv2.imshow(video.name,outputs['heatmap'])
                    # cv2.imshow(video.name, img)
                    # cv2.waitKey(1)
                    if cfg.COMMONSENSE.COMMONSENSE or cfg.DEFORMABLE.DEFORMABLE:
                        if cfg.COMMONSENSE.COMMONSENSE:
                            kernel_matrix=outputs['kernel_matrix']
                            search_matrix=outputs['search_matrix']
                            vis.image(kernel_matrix.transpose(2,0,1),win=kernelmatrix_win,opts=dict(title='kernel_matrix'))
                            vis.image(search_matrix.transpose(2,0,1),win=searchmatrix_win,opts=dict(title='search_matrix'))
                            r,g,b=cv2.split(search_matrix)
                            search_matrix=cv2.merge([b,g,r])
                            cv2.imwrite(save_image_path+'/matrixs/searchmatrix_{}.jpg'.format(idx),search_matrix)
                        sf=outputs['vis_ksf']
                        cf=outputs['vis_kcf']
                        vis.image(sf.transpose(2,0,1),win=kfeature_win,opts=dict(title='KSfeature'))
                        vis.image(cf.transpose(2,0,1),win=kcommonsense_win,opts=dict(title='KCfeature'))
                        sf=outputs['vis_ssf']
                        cf=outputs['vis_scf']
                        vis.image(sf.transpose(2,0,1),win=sfeature_win,opts=dict(title='SSfeature'))
                        vis.image(cf.transpose(2,0,1),win=scommonsense_win,opts=dict(title='SCfeature'))
                        original_heatmap=outputs['original_heatmap']
                        cv2.putText(original_heatmap, str(idx), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                        cv2.imwrite(save_image_path+'/original_heatmaps/original_heatmap_{}.jpg'.format(idx),original_heatmap)
                        b,g,r=cv2.split(original_heatmap)
                        original_heatmap=cv2.merge([r,g,b])
                        vis.image(original_heatmap.transpose(2,0,1),win=original_heatmap_win,opts=dict(title='original_heatmap'))
                        # embed()
                        # exit()
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    
    if os.path.isdir(args.snapshot):
        checkpoints=os.listdir(args.snapshot)
        for checkpoint in checkpoints:
            main(os.path.join(args.snapshot,checkpoint))
    else:
        main(args.snapshot)

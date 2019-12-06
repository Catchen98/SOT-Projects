import os
from collections import defaultdict
from os import path as osp
import argparse
from IPython import embed
import numpy as np
import torch
from cycler import cycler as cy
import json
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import motmetrics as mm

def parse_args():
    parser = argparse.ArgumentParser(description='test MOTA')
    parser.add_argument('--gt_json', default='/databack1/KITTI/kitti/tracking/training/kitti_val.json',help='gt json file')
    parser.add_argument('--result_json',default='/home/jn/codes/mmdetection/results/DBT_result.json',help='result json file')
    args = parser.parse_args()
    return args
def get_mot_accum(results, seqs):
    mot_accum = mm.MOTAccumulator(auto_id=True)    
    # embed()
    
    for i in range(len(seqs)-1):
        gt_boxes=[]
        gt_ids=[]
        if seqs[i]['ann']['bboxes']!=[]:
            for index,box in enumerate(seqs[i]['ann']['bboxes']):
                gt_boxes.append(box)
            for index,ids in enumerate(seqs[i]['ann']['track_id']):
                gt_ids.append(ids)
        else:
            continue

      
        gt_boxes = np.stack(gt_boxes, axis=0)
    

        gt_boxes = np.stack((gt_boxes[:, 0],
                                gt_boxes[:, 1],
                                gt_boxes[:, 2] - gt_boxes[:, 0],
                                gt_boxes[:, 3] - gt_boxes[:, 1]),
                            axis=1)

        track_ids=[]
        track_boxes=[]
        
        if results[i]['ann']['bboxes'] != []:
            for index,box in enumerate(results[i]['ann']['bboxes']):
                track_boxes.append(box)
            for index,ids in enumerate(results[i]['ann']['track_id']):
                track_ids.append(ids)
        else:
            continue

        track_boxes = np.stack(track_boxes, axis=0)
        
        track_boxes = np.stack((track_boxes[:, 0],
                                track_boxes[:, 1],
                                track_boxes[:, 2] - track_boxes[:, 0],
                                track_boxes[:, 3] - track_boxes[:, 1]),
                                axis=1)

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum

def evaluate_mot_accums(accums,names, generate_overall=False):
    # accums=[]
    # embed()
    # for i in range(len(results)-1):
    #     accums.append(get_mot_accum(results[i], seqs[i]))
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums, 
        metrics=mm.metrics.motchallenge_metrics, 
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names,)
    print(str_summary)
def main():
    args = parse_args()
    with open(args.result_json) as rf:
        results=json.load(rf)
    with open(args.gt_json) as gf:
        gts=json.load(gf)
    # embed()
    if len(results)!=len(gts):
        print('warning:the length of result not equal to the length of gt')
    result_videoids=[]
    gt_videoids=[]
    for i in range(len(results)-1):
        if results[i]['video_id'] not in result_videoids:
            result_videoids.append(results[i]['video_id'])
        if gts[i]['video_id'] not in gt_videoids:
            gt_videoids.append(gts[i]['video_id'])
    last_names=[]
    mot_accums=[]
    for video in result_videoids:
        
        if video not in gt_videoids:
            continue
        else:
            result=[]
            gt=[]
            for j in range(len(results)-1):
                if results[j]['video_id']==video:
                    result.append(results[j])
                if gts[j]['video_id']==video:
                    gt.append(gts[j])
            assert len(result)==len(gt), "the length of result {}video not equal to the length of gt {}video".format(video)
        # last_results.append(result)
        # last_gts.append(gt)
        last_names.append(video)
        # embed()
        mot_accums.append(get_mot_accum(result,gt))
    evaluate_mot_accums(mot_accums,last_names)
if __name__=='__main__':
    main()
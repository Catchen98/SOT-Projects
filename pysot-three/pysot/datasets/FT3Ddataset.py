# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from IPython import embed
from IO import read

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, flow, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name #dataset name
        self.root = root
        self.forward=os.path.join(flow,'into_future/')
        self.backward=os.path.join(flow,'into_past/')
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        videos=os.listdir(self.root)
        forwardflows=os.listdir(self.forward)
        backwardflows=os.listdir(self.backward)
        self.num = len(videos)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = videos
        self.forwardflows = forwardflows
        self.backwardflows=backwardflows
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video,frame):
        frame = "{:07d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame))
        return image_path
    def get_flow_anno(self, flow,frame,flow_path):
        frame = '{:07d}'.format(frame)
        flow_path = os.path.join(flow_path, flow,
                                 '{}.flo'.format(frame))
        return flow_path
    def get_positive_pair(self, index):
        video_name = self.videos[index]
        forwardflow_name=self.forwardflows[index]
        backwardflow_name=self.backwardflows[index]
        # video = self.labels[video_name]
        frames = os.listdir(os.path.join(self.root,video_name))
        template_frame = np.random.randint(0, len(frames)-1)
        search_frame=template_frame+1
        flow_frame=template_frame
        return self.get_image_anno(video_name, template_frame), \
            self.get_image_anno(video_name, search_frame),\
            self.get_flow_anno(forwardflow_name, flow_frame,self.forward),\
            self.get_flow_anno(backwardflow_name, flow_frame,self.backward)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self,):
        super(TrkDataset, self).__init__()

        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create anchor target
        self.anchor_target = AnchorTarget()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.FLOW,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
        neg=False
        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template_path, search_path, forwardflow_path, backwardflow_path = dataset.get_positive_pair(index)
        # get image
        template_image = cv2.imread(template_path)
        search_image = cv2.imread(search_path)
        forwardflow=read(forwardflow_path)
        backwardflow=read(backwardflow_path)
        
        # get bounding box
        # template_box = self._get_bbox(template_image, template[1])
        # search_box = self._get_bbox(search_image, search[1])

        # augmentation
        # template, _ = self.template_aug(template_image,
        #                                 template_box,
        #                                 cfg.TRAIN.EXEMPLAR_SIZE,
        #                                 gray=gray)

        # search, bbox = self.search_aug(search_image,
        #                                search_box,
        #                                cfg.TRAIN.SEARCH_SIZE,
        #                                gray=gray)

        # get labels
        # cls, delta, delta_weight, overlap = self.anchor_target(
        #         bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        template = template_image.transpose((2, 0, 1)).astype(np.float32)
        search = search_image.transpose((2, 0, 1)).astype(np.float32)
        return {
                'template': template,
                'search': search,
                'forwardflow': forwardflow,
                'backwardflow': backwardflow
                }

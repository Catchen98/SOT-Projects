# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import os
import cv2
import matplotlib.pyplot as plt
import torch
import pickle
from IPython import embed
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()

        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img, idx, video_name):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        #draw heatmap
        # heatmap_path = os.path.join('./test_results/' + video_name + '/heatmap/')
        # if not os.path.isdir(heatmap_path):
        #     os.makedirs(heatmap_path)

        # plt.imshow(pos_score, cmap='hot', interpolation='bilinear')
        # plt.savefig(heatmap_path +'pos_{}.jpg'.format(idx))
        #
        # if video_name:
        #     pos_heat = score.reshape(1, 5, 21, 21)
        #     pos_heat = torch.from_numpy(pos_heat)
        #     pos_heat, indicesn = torch.max(pos_heat, dim=1, keepdim=True)  # .reshape(17,17)
        #     neg_heat = 1.0 * torch.ones(1) - pos_heat
        #     pos_heat = F.interpolate(pos_heat, (287, 287), mode='bilinear',
        #                              align_corners=True).view(287, 287)
        #     neg_heat = F.interpolate(neg_heat, (287, 287),
        #                              mode='bilinear', align_corners=True).view(287, 287)
        #     pos_heat = np.float32(cv2.applyColorMap(np.uint8(255 * pos_heat.data.numpy()), cv2.COLORMAP_JET))
        #     neg_heat = np.float32(cv2.applyColorMap(np.uint8(255 * neg_heat.data.numpy()), cv2.COLORMAP_JET))
        #     # plt.savefig('./test_results/' + video_name + '/heatmap/pos_{}.jpg'.format(idx))
        #     x_crop = x_crop.squeeze().permute([1, 2, 0])
        #     x_crop = x_crop.data.cpu().numpy() / 255.0
        #     pos_heat_map = x_crop + pos_heat / 255
        #     if np.max(pos_heat_map) != 0:
        #         pos_heat_map /= np.max(pos_heat_map)
        #     pos_heat_map = np.uint8(pos_heat_map * 255)
        #     neg_heat_map = x_crop + neg_heat / 255
        #     if np.max(neg_heat_map) != 0:
        #         neg_heat_map /= np.max(neg_heat_map)
        #     neg_heat_map = np.uint8(neg_heat_map * 255)
        #     search_img_path = './test_results/' + video_name + '/heat_imgs/'
        #     if not os.path.exists(search_img_path):
        #         os.makedirs(search_img_path + '/pos/')
        #         os.makedirs(search_img_path + '/neg/')
        #     cv2.imwrite(search_img_path + '/pos/posheatmap_{}.jpg'.format(idx), pos_heat_map)
        #     cv2.imwrite(search_img_path + '/neg/negheatmap_{}.jpg'.format(idx), neg_heat_map)
        #     #save pos_features,neg_features
        #     search_feature = outputs['cls_feature']
        #     tem_features = outputs['tem_feature'].reshape(1,-1)
        #     search_feature = F.unfold(search_feature,(4,4)).reshape(1,256,4,4,21,21)
        #     pos_score = np.sum(np.reshape(score, [5, 21, 21]), axis=0)
        #     list_pos_score = pos_score.reshape(-1,1)
        #     score_sort=np.argsort(-list_pos_score,axis=0)
        #     maxxy = np.where(pos_score == list_pos_score[score_sort[0]])
        #     minxy = np.where(pos_score == list_pos_score[score_sort[-1]])
        #     middlexy = np.where(pos_score == list_pos_score[score_sort[2]])
        #     if len(maxxy[0]) !=1:
        #         maxxy=maxxy[0]
        #     if len(minxy[0]) !=1:
        #         minxy=minxy[0]
        #     if len(middlexy[0]) !=1:
        #         middlexy=middlexy[0]
        #     pos_feature = search_feature[:, :, :, :, maxxy[0], maxxy[1]].reshape(1,4096)
        #     neg_feature = search_feature[:, :, :, :, minxy[0], minxy[1]].reshape(1,4096)
        #     middle_feature = search_feature[:, :, :, :, middlexy[0], middlexy[1]].reshape(1,4096)
        #     if not os.path.exists('./test_results/' + video_name + '/save_features/pos_features'):
        #         os.makedirs('./test_results/' + video_name + '/save_features/pos_features/')
        #     if not os.path.exists('./test_results/' + video_name + '/save_features/neg_features'):
        #         os.makedirs('./test_results/' + video_name + '/save_features/neg_features/')
        #     if not os.path.exists('./test_results/' + video_name + '/save_features/middle_features'):
        #         os.makedirs('./test_results/' + video_name + '/save_features/middle_features/')
        #     pickle.dump(pos_feature.detach().cpu().numpy(), open('./test_results/' + video_name +
        #                                                          '/save_features/pos_features/pos_{}.pth'.format(idx), 'wb'))
        #     pickle.dump(neg_feature.detach().cpu().numpy(), open('./test_results/' + video_name +
        #                                                          '/save_features/neg_features/neg_{}.pth'.format(idx), 'wb'))
        #     pickle.dump(middle_feature.detach().cpu().numpy(), open('./test_results/' + video_name +
        #                                                          '/save_features/middle_features/middle_{}.pth'.format(idx),'wb'))
        #     if idx == 1:
        #         if not os.path.exists('./test_results/' + video_name + '/save_features/tem_features/'):
        #             os.makedirs('./test_results/' + video_name + '/save_features/tem_features/')
        #         pickle.dump(tem_features.detach().cpu().numpy(), open('./test_results/' + video_name +
        #                                                               '/save_features/tem_features/template.pth', 'wb'))
        # plt.show()
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import cv2
from flow_util import make_color_wheel, compute_color, flow_to_image


def set_id_grid(b, h, w):

    # b, _, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).float() # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).float() # [1, H, W]
    ones = torch.ones(1,h,w).float()
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    return pixel_coords


def inference(args, epoch, data_loader, model, offset=0):
    model.eval()
    if args.save_flow or args.render_validation:
        flow_folder = "{}/inference/{}.epoch-{}-flow-field".format(args.save, args.name.replace('/', '.'), epoch)
        if not os.path.exists(flow_folder):
            os.makedirs(flow_folder)

    args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches

    progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches),
                    desc='Inferencing ',
                    leave=True, position=offset)

    statistics = []
    total_loss = 0
    for batch_idx, (data, target) in enumerate(progress):
        if args.cuda:
            data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
        data, target = [Variable(d) for d in data], [Variable(t) for t in target]

        # when ground-truth flows are not available for inference_dataset,
        # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows,
        # depending on the type of loss norm passed in
        with torch.no_grad():
            losses, output = model(data[0], target[0], inference=True)

        losses = [torch.mean(loss_value) for loss_value in losses]
        loss_val = losses[0]  # Collect first loss for weight update
        total_loss += loss_val.item()
        loss_values = [v.item() for v in losses]

        # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
        loss_labels = list(model.module.loss.loss_labels)

        statistics.append(loss_values)
        # import IPython; IPython.embed()
        if args.save_flow or args.render_validation:
            for i in range(args.inference_batch_size):
                _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                flow_utils.writeFlow(join(flow_folder, '%06d.flo' % (batch_idx * args.inference_batch_size + i)),
                                     _pflow)

        progress.set_description(
            'Inference Averages for Epoch {}: '.format(epoch) + tools.format_dictionary_of_losses(loss_labels, np.array(
                statistics).mean(axis=0)))
        progress.update(1)

        if batch_idx == (args.inference_n_batches - 1):
            break

    progress.close()

    return

def compute_weight(self, embed_flow, embed_conv_feat):
    embed_flow_norm = mx.symbol.L2Normalization(data=embed_flow, mode='channel')
    embed_conv_norm = mx.symbol.L2Normalization(data=embed_conv_feat, mode='channel')
    weight = mx.symbol.sum(data=embed_flow_norm * embed_conv_norm, axis=1, keepdims=True)

    return weight

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, args, epoch, model_and_loss):
        batch_size, _, im_h, im_w = im_data.size()

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data


        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data[:, :3, :, :])  # B * 512 * H * W
        base_former_feat = self.RCNN_base(im_data[:, 3:6, :, :])  # B * 512 * H * W
        base_later_feat = self.RCNN_base(im_data[:, 6:9, :, :])  # B * 512 * H * W

        # argu initial images
        PIXEL_MEANS = cfg.PIXEL_MEANS.copy()
        PIXEL_MEANS = torch.from_numpy(PIXEL_MEANS).unsqueeze(0).float().cuda()
        now_img = (im_data[:, :3, :, :].permute(0, 2, 3, 1) + PIXEL_MEANS).permute(0, 3, 1, 2)
        former_img = (im_data[:, 3:6, :, :].permute(0, 2, 3, 1) + PIXEL_MEANS).permute(0, 3, 1, 2)
        later_img = (im_data[:, 6:9, :, :].permute(0, 2, 3, 1) + PIXEL_MEANS).permute(0, 3, 1, 2)

        ###%%% save former now later image %%%###
        # now_img_np = now_img.permute(0, 2, 3, 1)[0].cpu().numpy()
        # former_img_np = former_img.permute(0, 2, 3, 1)[0].cpu().numpy()
        # later_img_np = later_img.permute(0, 2, 3, 1)[0].cpu().numpy()
        # cv2.imwrite('915_now_img.png', now_img_np)
        # cv2.imwrite('915_former_img.png', former_img_np)
        # cv2.imwrite('915_later_img.png', later_img_np)
        ###%%% save former now later image %%%###

        # begin 1 / 4 method
        # 先将原始图像降采样为原来的 1 / 4
        grid_pixel = set_id_grid(batch_size, int(im_h / 4), int(im_w / 4)).cuda()
        u_grid = grid_pixel[0, 0, :, :].view(-1)
        u_norm = 2 * (u_grid / (int(im_w / 4) -1)) - 1
        v_grid = grid_pixel[0, 1, :, :].view(-1)
        v_norm = 2 * (v_grid / (int(im_h / 4) -1)) - 1
        pixel_coords = torch.stack([u_norm, v_norm], dim=1)
        pixel_coords_base_4 = pixel_coords.view(int(im_h / 4), int(im_w / 4), 2).repeat(batch_size, 1, 1, 1)

        grid_pixel = set_id_grid(batch_size, int(im_h / 16), int(im_w / 16)).cuda()
        u_grid = grid_pixel[0, 0, :, :].view(-1)
        u_norm = 2 * (u_grid / (int(im_w / 16) -1)) - 1
        v_grid = grid_pixel[0, 1, :, :].view(-1)
        v_norm = 2 * (v_grid / (int(im_h / 16) -1)) - 1
        pixel_coords = torch.stack([u_norm, v_norm], dim=1)
        pixel_coords_base_16 = pixel_coords.view(int(im_h / 16), int(im_w / 16), 2).repeat(batch_size, 1, 1, 1)

        now_img_sample = torch.nn.functional.grid_sample(now_img, pixel_coords_base_4, padding_mode='zeros')
        former_img_sample = torch.nn.functional.grid_sample(former_img, pixel_coords_base_4, padding_mode='zeros')
        later_img_sample = torch.nn.functional.grid_sample(later_img, pixel_coords_base_4, padding_mode='zeros')
        #
        former_sample_pair = torch.cat([now_img_sample.unsqueeze(2), former_img_sample.unsqueeze(2)], 2)
        later_sample_pair = torch.cat([now_img_sample.unsqueeze(2), later_img_sample.unsqueeze(2)], 2)
        # 得到的flow也是原来的 1 / 4
        former_flow = model_and_loss(former_sample_pair)
        later_flow = model_and_loss(later_sample_pair)
        # 中值滤波 。。 nearest
        former_flow_sample = torch.nn.functional.grid_sample(former_flow, pixel_coords_base_16, padding_mode='zeros')
        later_flow_sample = torch.nn.functional.grid_sample(later_flow, pixel_coords_base_16, padding_mode='zeros')
        # # end 1 / 4 method

        # now_img_np = now_img_sample.permute(0, 2, 3, 1)[0].cpu().numpy()
        # former_img_np = former_img_sample.permute(0, 2, 3, 1)[0].cpu().numpy()
        # later_img_np = later_img_sample.permute(0, 2, 3, 1)[0].cpu().numpy()
        # cv2.imwrite('916_now_sample_4.png', now_img_np)
        # cv2.imwrite('916_former_sample_4.png', former_img_np)
        # cv2.imwrite('916_later_sample_4.png', later_img_np)
        #

        # a = former_flow_sample[0].permute(1,2,0)
        # flow_image_former_np = flow_to_image(a)
        # cv2.imwrite('915_flow_former_4.png', flow_image_former_np)
        #
        # b = later_flow_sample[0].permute(1,2,0)
        # flow_image_later_np = flow_to_image(b)
        # cv2.imwrite('915_flow_later_4.png', flow_image_later_np)

        # # begin 1 / 16 methods
        # grid_pixel = set_id_grid(batch_size, int(im_h / 16), int(im_w / 16)).cuda()
        # u_grid = grid_pixel[0, 0, :, :].view(-1)
        # u_norm = 2 * (u_grid / (int(im_w / 16) - 1)) - 1
        # v_grid = grid_pixel[0, 1, :, :].view(-1)
        # v_norm = 2 * (v_grid / (int(im_h / 16) - 1)) - 1
        # pixel_coords = torch.stack([u_norm, v_norm], dim=1)
        # pixel_coords_base_16 = pixel_coords.view(int(im_h / 16), int(im_w / 16), 2).repeat(batch_size, 1, 1, 1)
        #
        # now_img_sample_16 = torch.nn.functional.grid_sample(now_img, pixel_coords_base_16, padding_mode='zeros')
        # former_img_sample_16 = torch.nn.functional.grid_sample(former_img, pixel_coords_base_16, padding_mode='zeros')
        # later_img_sample_16 = torch.nn.functional.grid_sample(later_img, pixel_coords_base_16, padding_mode='zeros')
        #
        # former_pair = torch.cat([now_img_sample_16.unsqueeze(2), former_img_sample_16.unsqueeze(2)], 2)
        # later_pair = torch.cat([now_img_sample_16.unsqueeze(2), later_img_sample_16.unsqueeze(2)], 2)
        # former_flow = model_and_loss(former_pair).detach()
        # later_flow = model_and_loss(later_pair).detach()
        #
        # former_flow_sample = torch.nn.functional.grid_sample(former_flow, pixel_coords_base_16, padding_mode='zeros') / 16
        # later_flow_sample = torch.nn.functional.grid_sample(later_flow, pixel_coords_base_16, padding_mode='zeros') / 16
        # # end 1 / 16 method

        # grid_pixel = set_id_grid(batch_size, int(im_h / 16), int(im_w / 16)).cuda()
        # u_grid = grid_pixel[0, 0, :, :].view(-1) + former_flow_sample[0, 0, :, :].view(-1)
        # u_norm_1 = 2 * (u_grid / 15) - 1
        # v_grid = grid_pixel[0, 1, :, :].view(-1) + former_flow_sample[0, 1, :, :].view(-1)
        # v_norm_1 = 2 * (v_grid / 15) - 1
        # pixel_coords = torch.stack([u_norm_1, v_norm_1], dim=1).view(int(im_h / 16), int(im_w / 16), 2)
        # base_former_feat_warped = torch.nn.functional.grid_sample(base_former_feat[[0]], pixel_coords.unsqueeze(0), padding_mode='zeros')
        # tempc = base_former_feat_warped[0][-100]
        # temp_d = ((tempc - tempc.min()) / (tempc.max() - tempc.min())) * 255
        # cv2.imwrite('915_debug.png', temp_d.cpu().detach().numpy())

        temp = 2 * (former_flow_sample) # 这里不要 -1
        u_norm = temp[:, 0, :, :].view(batch_size, -1) / (int(im_w / 16) - 1)
        v_norm = temp[:, 1, :, :].view(batch_size, -1) / (int(im_h / 16) - 1)
        pixel_coords_bias = torch.stack([u_norm, v_norm], dim=2)
        pixel_coords_bias_4 = pixel_coords_bias.view(batch_size, int(im_h / 16), int(im_w / 16), 2)
        pixel_coords_former2now_4 = pixel_coords_base_16 + pixel_coords_bias_4
        # base_later_feat_warped 2 * 512 * 16 * 16
        # 降采样。。。
        base_former_feat_warped = torch.nn.functional.grid_sample(base_former_feat, pixel_coords_former2now_4, padding_mode='zeros')

        temp = 2 * (later_flow_sample)
        u_norm = temp[:, 0, :, :].view(batch_size, -1) / (int(im_w / 16) - 1)
        v_norm = temp[:, 1, :, :].view(batch_size, -1) / (int(im_h / 16) - 1)
        pixel_coords_bias = torch.stack([u_norm, v_norm], dim=2)
        pixel_coords_bias_4 = pixel_coords_bias.view(batch_size, int(im_h / 16), int(im_w / 16), 2)
        pixel_coords_later2now_4 = pixel_coords_base_16 + pixel_coords_bias_4
        # base_later_feat_warped 2 * 512 * 16 * 16
        base_later_feat_warped = torch.nn.functional.grid_sample(base_later_feat, pixel_coords_later2now_4, padding_mode='zeros')

        base_current_feat_warped_embedding = self.get_embednet(base_feat)  # 1 * 2048 * 16 * 16
        base_former_feat_warped_embedding = self.get_embednet(base_former_feat_warped)  # 1 * 2048 * 16 * 16
        base_later_feat_warped_embedding = self.get_embednet(base_later_feat_warped)  # 1 * 2048 * 16 * 16

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        former_weight = cos(base_current_feat_warped_embedding, base_former_feat_warped_embedding).unsqueeze(1)
        later_weight = cos(base_current_feat_warped_embedding, base_later_feat_warped_embedding).unsqueeze(1)

        # softmax opperation
        former_weight_tile = former_weight.repeat(1, 512, 1, 1)  # repeat in channle
        later_weight_tile = later_weight.repeat(1, 512, 1, 1)  # repeat in channle
        former_weight_tile_softmax = F.softmax(former_weight_tile, dim=0)
        later_weight_tile_softmax = F.softmax(later_weight_tile, dim=0)
        aggregated_feat = base_former_feat_warped * former_weight_tile_softmax + base_later_feat_warped * later_weight_tile_softmax

        # # 降采样了 16 的特征
        # tempa = base_former_feat[0][-100]
        # temp_b = ((tempa - tempa.min()) / (tempa.max() - tempa.min())) * 255
        # cv2.imwrite('915_feature_base_former_feat_100.png', temp_b.cpu().detach().numpy())
        # # 降采样了 16 的特征
        # tempa = base_feat[0][-100]
        # temp_b = ((tempa - tempa.min()) / (tempa.max() - tempa.min())) * 255
        # cv2.imwrite('915_feature_base_feat_100.png', temp_b.cpu().detach().numpy())
        # # 降采样了 16 的特征
        # tempa = base_later_feat[0][-100]
        # temp_b = ((tempa - tempa.min()) / (tempa.max() - tempa.min())) * 255
        # cv2.imwrite('915_feature_base_later_feat_100.png', temp_b.cpu().detach().numpy())
        # # 从降采样了 16 的特征 进行升采样到 64 * 224
        # tempc = base_former_feat_warped[0][-100]
        # temp_d = ((tempc - tempc.min()) / (tempc.max() - tempc.min())) * 255
        # cv2.imwrite('915_feature_base_former_warped_feat_100_verify_4.png', temp_d.cpu().detach().numpy())
        # # 从降采样了 16 的特征 进行升采样到 64 * 224
        # tempc = base_later_feat_warped[0][-100]
        # temp_d = ((tempc - tempc.min()) / (tempc.max() - tempc.min())) * 255
        # cv2.imwrite('915_feature_base_later_warped_feat_100_verify_4.png', temp_d.cpu().detach().numpy())
        # # 获取的光流是原始图像的大小
        # a = former_flow[0].permute(1,2,0)
        # flow_image_former_np = flow_to_image(a)
        # cv2.imwrite('915_flow_former_1.png', flow_image_former_np)
        # # 获取的光流是原始图像的大小
        # b = later_flow[0].permute(1,2,0)
        # flow_image_later_np = flow_to_image(b)
        # cv2.imwrite('915_flow_later_1.png', flow_image_later_np)
        # #
        # a = former_flow_sample[0].permute(1,2,0)
        # flow_image_former_np = flow_to_image(a)
        # cv2.imwrite('915_flow_former_sample_4.png', flow_image_former_np)
        #
        # b = later_flow_sample[0].permute(1,2,0)
        # flow_image_later_np = flow_to_image(b)
        # cv2.imwrite('915_flow_later_sample_4.png', flow_image_later_np)

        # print('faster rcnn ning debug')
        # from IPython import embed
        # embed()

        # # warp image to verify
        # grid_pixel = set_id_grid(batch_size, 256, 256).float().cuda()
        # u_grid = grid_pixel[0, 0, :, :].view(-1) + former_flow[0][0].view(-1)
        # u_norm = 2 * (u_grid / 255) - 1
        # v_grid = grid_pixel[0, 1, :, :].view(-1) + former_flow[0][1].view(-1)
        # v_norm = 2 * (v_grid / 255) - 1
        # pixel_coords = torch.stack([u_norm, v_norm], dim=1)
        # pixel_coords_later = pixel_coords.view(256, 256, 2)
        # aa = torch.nn.functional.grid_sample(former_img[[0], :, :, :], pixel_coords_later.unsqueeze(0), padding_mode='zeros')
        # aaa = aa[0].permute(1,2,0).cpu().detach().numpy()
        # cv2.imwrite('aaa.png', aaa)
        #
        # grid_pixel = set_id_grid(im_data)
        # u_grid = grid_pixel[0, 0, :, :].view(-1) + later_flow[0][0].view(-1)
        # u_norm = 2 * (u_grid / 255) - 1
        # v_grid = grid_pixel[0, 1, :, :].view(-1) + later_flow[0][1].view(-1)
        # v_norm = 2 * (v_grid / 255) - 1
        # pixel_coords = torch.stack([u_norm, v_norm], dim=1)
        # pixel_coords_later = pixel_coords.view(256, 256, 2)
        # bb = torch.nn.functional.grid_sample(later_img[[0], :, :, :], pixel_coords_later.unsqueeze(0), padding_mode='zeros')
        # bbb = bb[0].permute(1,2,0).cpu().detach().numpy()
        # cv2.imwrite('bbb.png', bbb)

        # flownet and warpping process

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(aggregated_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-12-11 16:41:28  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-12-11 16:41:28

import matplotlib.pyplot as plt
import numpy as np
import cv2
import mmcv
import matplotlib
def kernel_inv_map(vis_attr, target_point, map_h, map_w):
    #-1,0,1
    pos_shift = [vis_attr['dilation'] * 0 - vis_attr['pad'],
                 vis_attr['dilation'] * 1 - vis_attr['pad'],
                 vis_attr['dilation'] * 2 - vis_attr['pad']]
    source_point = []
    for idx in range(vis_attr['filter_size']**2):
        # -1,0,1  -1,-1,-1
        # -1,0,1   0, 0, 0
        # -1,0,1   1, 1, 1
        #grid,y,x
        cur_source_point = np.array([target_point[0] + pos_shift[idx // 3],
                                     target_point[1] + pos_shift[idx % 3]])
        if cur_source_point[0] < 0 or cur_source_point[1] < 0 \
                or cur_source_point[0] > map_h - 1 or cur_source_point[1] > map_w - 1:
            continue
        source_point.append(cur_source_point.astype('f'))
    return source_point

def offset_inv_map(source_points, offset):
    for idx, _ in enumerate(source_points):
        source_points[idx][0] += offset[2*idx]
        source_points[idx][1] += offset[2*idx + 1]
        # print(np.min(offset),np.max(offset))
    return source_points

def get_bottom_position(vis_attr, top_points, all_offset):
    map_h = all_offset[0].shape[2]
    map_w = all_offset[0].shape[3]
    for level in range(vis_attr['plot_level']):
        source_points = []
        for idx, cur_top_point in enumerate(top_points):
            cur_top_point = np.round(cur_top_point)
            if cur_top_point[0] < 0 or cur_top_point[1] < 0 \
                or cur_top_point[0] > map_h-1 or cur_top_point[1] > map_w-1:
                continue
            cur_source_point = kernel_inv_map(vis_attr, cur_top_point, map_h, map_w)
            cur_offset = np.squeeze(all_offset[level][:, :, int(cur_top_point[0]), int(cur_top_point[1])])
            #print(all_offset[level][:, :, int(cur_top_point[0]), int(cur_top_point[1])])
            # print(cur_offset.shape)
            cur_source_point = offset_inv_map(cur_source_point, cur_offset)
            source_points = source_points + cur_source_point
            # print(cur_source_point)
        top_points = source_points
    return source_points

def plot_according_to_point(vis_attr, im, source_points, map_h, map_w, color=[255,0,0]):
    plot_area = vis_attr['plot_area']
    for idx, cur_source_point in enumerate(source_points):
        # print(im.shape[0] / map_h)
        y = np.round((cur_source_point[0] + 0.5) * im.shape[0] / map_h).astype('i')
        x = np.round((cur_source_point[1] + 0.5) * im.shape[1] / map_w).astype('i')

        if x < 0 or y < 0 or x > im.shape[1]-1 or y > im.shape[0]-1:
            continue
        y = min(y, im.shape[0] - vis_attr['plot_area'] - 1)
        x = min(x, im.shape[1] - vis_attr['plot_area'] - 1)
        y = max(y, vis_attr['plot_area'])
        x = max(x, vis_attr['plot_area'])

        im[y-plot_area:y+plot_area+1, x-plot_area:x+plot_area+1, :] = np.tile(
            np.reshape(color, (1, 1, 3)), (2*plot_area+1, 2*plot_area+1, 1)
        )
    return im



def show_dconv_offset(im, all_offset, step=[2, 2], filter_size=3,
                      dilation=1, pad=1, plot_area=1, plot_level=4,stride=8):
    vis_attr = {'filter_size': filter_size, 'dilation': dilation, 'pad': pad,
                'plot_area': plot_area, 'plot_level': plot_level,'stride':stride}

    map_h = all_offset[0].shape[2]
    map_w = all_offset[0].shape[3]

    step_h = step[0]
    step_w = step[1]
    start_h = np.round(step_h / 2).astype(np.int)
    start_h=15*2
    start_w = np.round(step_w / 2).astype(np.int)
    start_w=20*2
    plt.figure(figsize=(15, 5))
    for im_h in range(start_h, map_h, step_h):
        for im_w in range(start_w, map_w, step_w):
            target_point = np.array([im_h, im_w]).astype(np.int)
            source_y = np.round(target_point[0] * im.shape[0] / map_h).astype(np.int)
            source_x = np.round(target_point[1] * im.shape[1] / map_w).astype(np.int)
            if source_y < plot_area or source_x < plot_area \
                    or source_y >= im.shape[0] - plot_area or source_x >= im.shape[1] - plot_area:
                print('out of image')
                continue

            cur_im = np.copy(im)
            source_points = get_bottom_position(vis_attr, [target_point], all_offset)
            cur_im = plot_according_to_point(vis_attr, cur_im, source_points, map_h, map_w)
            
            cur_im[source_y-3:source_y+3+1, source_x-3:source_x+3+1, :] = \
                np.tile(np.reshape([0, 255, 0], (1, 1, 3)), (2*3+1, 2*3+1, 1))

            print('showing',im_h,im_w)
            plt.axis("off")
            plt.imshow(cur_im)
            plt.show(block=False)
            plt.pause(0.01)
            plt.clf()
if __name__=='__main__':
    img=matplotlib.image.imread('/home/ld/RepPoints/offset/000100.png')
    # height, width = img.shape[:2]
    # size=(int(width/8),int(height/8))
    # img=cv2.resize(img,size)

    offset156=np.load('/home/ld/RepPoints/offset/resnetl20.npy')
    offset78=np.load('/home/ld/RepPoints/offset/resnetl21.npy')
    offset39=np.load('/home/ld/RepPoints/offset/resnetl22.npy')
    offset20=np.load('/home/ld/RepPoints/offset/resnetl23.npy')
    print(offset156.shape)
    # offset10=np.load('/home/ld/RepPoints/offset/init10.npy')
    show_dconv_offset(img,[offset20,offset39,offset78,offset156])
    # show_dconv_offset(im, [res5c_offset, res5b_offset, res5a_offset])
    # #detach the init grad
    # pts_out_refine = pts_out_refine + pts_out_init.detach()
    # if dcn_offset.shape[-1]==156:
    #     np.save('/home/ld/RepPoints/offset/init.npy',dcn_offset.data.cpu().numpy())
    #     np.save('/home/ld/RepPoints/offset/refine.npy',pts_out_refine.data.cpu().numpy())
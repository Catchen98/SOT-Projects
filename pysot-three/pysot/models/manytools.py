import torch
import numpy as np
import pickle
import cv2
import os
import torch.nn.functional as F
from IPython import embed

def compute_argmaxxcorr(xcorr):
    score=xcorr.reshape(1,2,-1).permute(0,2,1)
    score=F.softmax(score,dim=2)[0,:,1].view(1,5,17,17)
    score,indicesn=torch.max(score,dim=1,keepdim=True)
    score=score.data.cpu().numpy()
    index=np.argmax(score)
    index=np.unravel_index(index,(score.shape[2],score.shape[3]))
    return index
def save_features(save_path,features,index=[]):
    if index:
        pickle.dump(features[:,:,index[0]:index[0]+4,index[1]:index[1]+4].reshape(1, -1).detach().cpu().numpy(), 
                                open(save_path, 'wb'))
    else:
        pickle.dump(features.reshape(1, -1).detach().cpu().numpy(),open(save_path, 'wb'))
def get_meshgrid(B,H,W):
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W)#.repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W)#.repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().cuda()
    grid = grid.repeat(B,1,1,1)
    return grid
def draw_heatmap(inputs,image,rgb=True):
    size=image.shape[-1]
    inputs=F.interpolate(inputs,(size,size),mode='bilinear',align_corners=True).view(size,size)
    # embed()
    heatmap=np.float32(cv2.applyColorMap(np.uint8(255*inputs.data.cpu().numpy()),cv2.COLORMAP_JET))
    heatmap=heatmap/255+(image/255.0).permute(1,2,0).data.cpu().numpy()
    heatmap/=np.max(heatmap)
    heatmap=np.uint8(heatmap*255)
    if rgb:
        b,g,r=cv2.split(heatmap)
        heatmap=cv2.merge([r,g,b])
    return heatmap
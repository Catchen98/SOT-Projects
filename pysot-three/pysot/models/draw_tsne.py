import pickle
import os
import sys
import numpy as np
import cv2
import argparse
from fire import Fire
from matplotlib import pyplot as plt
from tsnecuda import TSNE
from tqdm import tqdm

from IPython import embed


def main():
    parser = argparse.ArgumentParser(description='draw feature tsne results')
    parser.add_argument('--video_name', default='',type=str,help='video name')
    parser.add_argument('--model_name',default='',type=str,help='model name')
    args = parser.parse_args()

    root_path = '/home/jn/codes/pysot-master/results/save_features/'+args.model_name+'/'+args.video_name
    save_path = root_path + '/tsne_results/'
    box_path = '/home/jn/codes/pysot-master/results/save_image/'+args.model_name+'/'+args.video_name+'/images/'
    heatmap_path = '/home/jn/codes/pysot-master/results/save_image/'+args.model_name+'/'+args.video_name+'/heatmaps/'
    originalheatmap_path = '/home/jn/codes/pysot-master/results/save_image/'+args.model_name+'/'+args.video_name+'/original_heatmaps/'
    # matrix_path = '/home/jn/codes/pysot-master/results/save_image/'+args.model_name+'/'+args.video_name+'/matrixs/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    searchsf_path = root_path + '/search_sfeatures/'
    templatesf_path = root_path + '/template_sfeatures/'
    searchcf_path = root_path + '/search_cfeatures/'
    templatecf_path = root_path + '/template_cfeatures/'
    
    # template_sfs=os.listdir(templatesf_path)
    # search_sfs = os.listdir(searchsf_path)
    # template_cfs = os.listdir(templatecf_path)
    # search_cfs = os.listdir(searchcf_path)
    # # filenames.sort(key=lambda x:int(x[4:-4]))
    # # print(np.array(filenames[:10]))
    # search_cfeature_all = []
    # search_sfeature_all = []
    # # neg_feature_all = []
    # # middle_feature_all = []
    # feature_result_all = []
    # for template_cf in template_cfs:
    #     tem_cfeature = pickle.load(open(templatecf_path+template_cf, 'rb'))
    #     feature_result_all += tem_cfeature.tolist()[0]
    # for template_sf in template_sfs:
    #     tem_sfeature = pickle.load(open(templatesf_path+template_sf, 'rb'))
    #     feature_result_all += tem_sfeature.tolist()[0]
    # # embed()
    # for search_cf in search_cfs:
    #     search_cfeature = pickle.load(open(searchcf_path+search_cf, 'rb'))
    #     search_cfeature = search_cfeature.reshape(-1)
    #     search_cfeature_all += search_cfeature.tolist()#[:256]
    # feature_result_all.extend(search_cfeature_all)
    # for search_sf in search_sfs:
    #     search_sfeature = pickle.load(open(searchsf_path+search_sf, 'rb'))
    #     search_sfeature = search_sfeature.reshape(-1)
    #     search_sfeature_all += search_sfeature.tolist()
    # # embed()
    # feature_result_all.extend(search_sfeature_all)
    # for idx in range(1,len(middlenames)+1):
    #     middle_feature = pickle.load(open(middle_path+'middle_{}.pth'.format(idx), 'rb'))
    #     middle_feature = middle_feature.reshape(-1)
    #     middle_feature_all += middle_feature.tolist()
    #     # neg_feature_all.append(neg_feature)
    # feature_result_all.extend(middle_feature_all)
    # embed()
    # feature_result_all = np.array(feature_result_all).reshape(len(search_sfs)*2+2,-1)

    # point2D = TSNE().fit_transform(feature_result_all)
    # x_min = point2D[:, 0].min()
    # x_max = point2D[:, 0].max()
    # y_min = point2D[:, 1].min()
    # y_max = point2D[:, 1].max()
    # point2D_normal = (point2D - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
    # color_search = [[[0, 0, x]] for x in np.array(range(len(searchnames))) / len(searchnames) * 1]
    # color_search = [[[1, 1, x]] for x in np.array(range(len(negnames))) / len(negnames) * 1]
    # embed()
    box_imgs = os.listdir(box_path)
    # for idx in tqdm(np.array(range(0,len(search_sfs)))+1):
    for idx in tqdm(np.array(range(0,len(box_imgs)))+1):
        fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(221)
        #plt.figure()
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
     
        # for i in range(idx):
        #     plt.scatter(point2D_normal[0, 0], point2D_normal[0, 1], c='r',marker='o')#r,o
        #     plt.scatter(point2D_normal[1, 0], point2D_normal[1, 1], c='m',marker='*')
        #     plt.scatter(point2D_normal[i + 2, 0],
        #                 point2D_normal[i + 2, 1], c='c',marker='x')#c,x
        #     plt.scatter(point2D_normal[i + 2 + len(search_cfs)-1, 0],
        #                 point2D_normal[i + 2 + len(search_cfs)-1, 1], c='y',marker='^')
        box_img = cv2.imread(box_path+'image_{}.jpg'.format(idx))
        heatmap_img = cv2.imread(heatmap_path+'heatmap_{}.jpg'.format(idx))
        # originalheatmap_img = cv2.imread(originalheatmap_path+'original_heatmap_{}.jpg'.format(idx))
        # matrix_img = cv2.imread(matrix_path + 'searchmatrix_{}.jpg'.format(idx))
        # if idx>=10:
        bx = fig.add_subplot(121)
        plt.imshow(box_img[:, :, ::-1])
        dx= fig.add_subplot(122)
        plt.imshow(heatmap_img[:, :, ::-1])
        # cx = fig.add_subplot(224)
        # plt.imshow(matrix_img[:, :, ::-1])
        # plt.imshow(originalheatmap_img[:,:,::-1])

        plt.savefig(save_path + str(idx) + '.jpg')
    plt.close()


if __name__ == '__main__':
    Fire(main)

import cv2
import argparse
import os
from IPython import embed


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='save video')
    parser.add_argument('--video_name','-vn',dest='video_name')
    parser.add_argument('--model_name',type=str, help='model name')
    args = parser.parse_args()

    root_path='/home/jn/codes/pysot-master/results/save_features/'
    img_path = root_path+args.model_name+'/'+args.video_name+'/tsne_results/'
    # root_path='/home/jn/codes/pysot-master/results/save_image/'
    # img_path = root_path+args.model_name+'/'+args.video_name+'/images/'
    save_video=root_path+args.model_name+'/'+args.video_name+'/'
    video_writer = cv2.VideoWriter(save_video+args.model_name+'_'+args.video_name+'_video.avi',
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,(800,800))
    img_names = os.listdir(img_path)
    for idx in range(1,len(img_names)+1):
        img = cv2.imread(img_path+'{}.jpg'.format(idx))
        video_writer.write(img)

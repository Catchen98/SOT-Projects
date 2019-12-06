import argparse
import os
import sys
import time
import os.path as osp
import shutil
import tempfile
import json
from tqdm import tqdm
from IPython import embed
from sacred import Experiment
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
# print(os.getcwd())
# sys.path.append('./mmdetection')
# from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
# sys.path.append('./mottracker/src/')



def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--tracker',help='type of tracker')
    # parser.add_argument('--out', help='output result file')
    # parser.add_argument(
    #     '--json_out',
    #     help='output result file name without extension',
    #     type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    # parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
class LoadImage(object):
    
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
def main():
    args = parse_args()
    data_path = '/databack1/KITTI/kitti/tracking'
    json_name = 'training/kitti_val.json'
    
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    # print('test')
    # embed()
    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # exit()
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    if args.tracker=='det':
        from mottracker.src.tracktor.tracker_det import Tracker
    elif args.tracker=='pred':
        from mottracker.src.tracktor.tracker_pred import Tracker
    else:
        from mottracker.src.tracktor.tracker_detpred import Tracker
    
    tracker = Tracker(model, cfg.tracker_cfg)
    time_total = 0
    num_frames = 0
    mot_accums = []
    eval_data=[]
    
    start = time.time()

    # log.info(f"Tracking")
    with open(os.path.join(data_path,json_name),'r',encoding='utf-8') as f:
	    datas=json.load(f)
    prevideo_name=None
    eval_data=[]
    for i, image in enumerate(tqdm(data_loader)):
        # print(i,'in',len(datas))
        video_name=image['img_meta'][0].data[0][0]['filename'].split('/')[-2]
        image_name=image['img_meta'][0].data[0][0]['filename'].split('/')[-1]
        if video_name != prevideo_name:
            tracker.reset()#####初始化
            frame_data={}
            num_frames=i
        with torch.no_grad():
            tracker.step(image)####跟踪
        results = tracker.get_results()
        # print('results:',len(results))

        bboxes=[]
        labels=[]
        ids=[]
        for key1 in results.keys():
            for key2 in results[key1]:
                if key2==i-num_frames:
                    bboxes.append(results[key1][key2][:4].tolist())
                    labels.append(1)
                    ids.append(key1)
        # embed()
        frame_data={"video_id":video_name,"filename":'training/image_02/'+video_name+'/'+image_name,
                    "ann":{"bboxes":bboxes,"labels":labels,
                           "track_id":ids}}
        eval_data.append(frame_data)
        prevideo_name=video_name
        # if i==5:
        #     break
    with open(os.path.join('./results','DBT_{}result.json'.format(args.tracker)),'w',encoding='utf-8') as f:
        data=json.dump(eval_data,f)
    # time_total += time.time() - start

if __name__ == '__main__':
    main()

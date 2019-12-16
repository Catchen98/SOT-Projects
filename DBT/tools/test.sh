#!bash
python tools/test.py configs/kitti_finetunefasterrcnn101.py \
    snapshot/r101/epoch_50.pth \
    --show
# python tools/test.py configs/kitti_test.py \
#     checkpoints/faster_rcnn_r101_fpn_1x_20181129-d1468807.pth \
#     --show
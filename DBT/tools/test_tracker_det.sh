python tools/test_tracker.py --config configs/kitti_test.py \
    --checkpoint checkpoints/faster_rcnn_r101_fpn_1x_20181129-d1468807.pth \
    --tracker 'det'
# python tools/test_tracker.py --config configs/kitti_test.py \
#     --checkpoint snapshot/r101/epoch_20.pth \
#     --tracker 'det'
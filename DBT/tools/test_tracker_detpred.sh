python tools/test_tracker.py --config configs/reppoints_moment_r101_dcn_fpn_kitti.py \
    --checkpoint snapshot/reppoint.pth \
    --tracker 'det_pred'
# python tools/test_tracker.py --config configs/kitti_test.py \
# --checkpoint snapshot/r101/epoch_40.pth \
# --tracker 'det_pred'
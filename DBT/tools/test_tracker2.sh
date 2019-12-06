#!bash
source activate open-mmlab
python tools/test_tracker2.py configs/kitti_test.py \
    snapshot/r101/epoch_40.pth \
    --show
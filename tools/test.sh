WORK_DIR=$1
BEST_EPOCH=$2

python tools/test_custom.py configs/widerface/$WORK_DIR.py work_dirs/$WORK_DIR/epoch_50.pth --eval mAP

python tools/test_custom.py configs/widerface/$WORK_DIR.py work_dirs/$WORK_DIR/epoch_80.pth --eval mAP

python tools/test_custom.py configs/widerface/$WORK_DIR.py work_dirs/$WORK_DIR/epoch_$BEST_EPOCH.pth --eval mAP

python tools/test_custom.py configs/widerface/$WORK_DIR.py work_dirs/$WORK_DIR/epoch_130.pth --eval mAP
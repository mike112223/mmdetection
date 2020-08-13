nohup python tools/test_custom.py configs/widerface/retinaface_fulltrain_data+aug_ignore9_nosub1_anchor_tta5.py work_dirs/retinaface_fulltrain_data+aug_ignore9_nosub1_anchor/epoch_101.pth --eval mAP > test5.log &

nohup python tools/test_custom.py configs/widerface/retinaface_fulltrain_data+aug_ignore9_nosub1_anchor_tta5_flip.py work_dirs/retinaface_fulltrain_data+aug_ignore9_nosub1_anchor/epoch_101.pth --eval mAP > testf5.log &

nohup python tools/test_custom.py configs/widerface/retinaface_fulltrain_data+aug_ignore9_nosub1_anchor_tta.py work_dirs/retinaface_fulltrain_data+aug_ignore9_nosub1_anchor/epoch_101.pth --eval mAP > test.log &

#inference bash

# generate test json file
python tools/post_process/txt2coco.py

python tools/generate_testA.py

chmod +x tools/dist_test.sh

mkdir -p /data/user_data/results_testa_submit
# model 1
./tools/dist_test.sh configs/ship/cascade_convnext_base_large_scale_onlytraindata_noflip_anchor4_ratio.py /data/user_data/work_dirs/convnext.pth 4 --format-only --eval-options "jsonfile_prefix=/data/user_data/results_testa_submit/convnext_submit"

#model 2 mosaic+close
./tools/dist_test.sh configs/ship/cascade_convnext_base_large_scale_onlytraindata_noflip_anchor4_ratio_mosaic_close.py /data/user_data/work_dirs/convnext_mosaic.pth 4 --format-only --eval-options "jsonfile_prefix=/data/user_data/results_testa_submit/convnext_mosaic_submit"

#model 3 mosaic+close
./tools/dist_test.sh configs/ship/cascade_swin_small_large_scale_onlytraindata_noflip_anchor4_ratio_mosaic_close.py /data/user_data/work_dirs/swin_mosaic.pth 4 --format-only --eval-options "jsonfile_prefix=/data/user_data/results_testa_submit/swin_mosaic_submit"

python tools/post_process/wbf.py

python tools/post_process/json2submit.py
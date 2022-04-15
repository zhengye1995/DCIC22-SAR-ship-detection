
# training bash

chmod +x tools/dist_train.sh
#generate coco json for training
python tools/post_process/txt2coco.py

# model 1
./tools/dist_train.sh configs/ship/cascade_convnext_base_large_scale_onlytraindata_noflip_anchor4_ratio.py 4 --no-validate

#model 2 mosaic+close
./tools/dist_train.sh configs/ship/cascade_convnext_base_large_scale_onlytraindata_noflip_anchor4_ratio_mosaic.py 4 --no-validate

./tools/dist_train.sh configs/ship/cascade_convnext_base_large_scale_onlytraindata_noflip_anchor4_ratio_mosaic_close.py 4 --no-validate

#model 3 mosaic+close
./tools/dist_train.sh configs/ship/cascade_swin_small_large_scale_onlytraindata_noflip_anchor4_ratio_mosaic.py 4 --no-validate

./tools/dist_train.sh configs/ship/cascade_swin_small_large_scale_onlytraindata_noflip_anchor4_ratio_mosaic_close.py 4 --no-validate

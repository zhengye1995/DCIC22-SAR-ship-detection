# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm
from glob import glob
from PIL import Image

if __name__ == '__main__':
    print("generate testA json...")
    with open('/data/user_data/annotations/train.json', 'r') as f:
        categories = json.load(f)['categories']
    img_root = '/data/raw_data/test_dataset/测试集/'
    save_path_root = "/data/user_data/annotations/"
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1
    for imgpath in tqdm(glob(img_root + "/*")):
        w, h = Image.open(imgpath).size
        imgname = os.path.basename(imgpath)
        images.append({
            "id": image_id,
            "height": h,
            "width": w,
            "file_name": imgname
        })
        annotations.append({
            "id": annotation_id,
            "bbox": [0,0,20,20],
            "area": 400.0,
            "segmentation": [[]],
            "iscrowd": 0,
            "category_id": 1,
            "image_id": image_id
        })
        image_id+=1
        annotation_id+=1
    instance = {"images": images, "annotations": annotations, "categories": categories}
    with open(save_path_root + 'testA.json', 'w') as f:
        json.dump(instance, f, indent=1)



import json
import os
from glob import glob
from tqdm import tqdm

def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1],
            points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]

def yolo2coco(box, img_h=256, img_w=256):
    x_center, y_center, w, h = box
    x_center = x_center * img_w
    y_center = y_center * img_h
    w = w * img_w
    h = h * img_h
    xmin = max(0, x_center - w/2)
    ymin = max(0, y_center - h/2)
    return [xmin, ymin, w, h]

images = []
annotations = []
categories = [{
    "id": 1,
    "name": "ship"
}]
image_id = 1
anno_id = 1
for txt_file in tqdm(glob('/data/raw_data/training_dataset/A/*.txt')):
    images.append({
        "id": image_id,
        "file_name": os.path.splitext(os.path.basename(txt_file))[0] + '.jpg',
        "height": 256,
        "width": 256
    })
    with open(txt_file, 'r') as f:
        annos = f.readlines()
    for anno in annos:
        _, x_center, y_center, w, h = anno.strip().split(' ')
        x_center = float(x_center)
        y_center = float(y_center)
        w = float(w)
        h = float(h)
        box = yolo2coco([x_center, y_center, w, h])

        annotations.append({
            "id": anno_id,
            "image_id": image_id,
            "bbox": box,
            "category_id": 1,
            "area": box[2] * box[3],
            "iscrowd": 0,
            "ignore": 0,
            "segmentation": get_segmentation(box)
        })
        anno_id += 1
    image_id += 1

final_instance = {"images": images, "annotations": annotations, "categories": categories}
os.makedirs('/data/user_data/annotations', exist_ok=True)
with open('/data/user_data/annotations/train.json', 'w') as f:
    json.dump(final_instance, f, indent=1)




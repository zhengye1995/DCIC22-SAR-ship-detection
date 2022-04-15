import json
import os
from glob import glob
test_json = '/data/user_data/annotations/testA.json'
bbox_results = '/data/user_data/results_testa_submit/wbf.json'

with open(test_json, 'r') as f:
    test_info = json.load(f)

with open(bbox_results, 'r') as f:
    bbox_anno = json.load(f)


imageid2annos = {}
imageid2name = {}
for image in test_info['images']:
    image_id = image['id']
    imageid2name[image_id] = image['file_name']
    if image_id not in imageid2annos:
        imageid2annos[image_id] = []

os.makedirs('/data/prediction_result', exist_ok=True)
submit_file = '/data/prediction_result/result.csv'


SCORE_THR = 0.95

for anno in bbox_anno:
    # if anno['score'] > SCORE_THR:
    imageid2annos[anno['image_id']].append(anno)



all_img = []
for img in glob('/data/raw_data/test_dataset/测试集/*'):
    all_img.append(os.path.splitext(os.path.basename(img))[0])

with open(submit_file, 'w') as fp:
    for imageid, imagename in imageid2name.items():
        all_anno = os.path.splitext(os.path.basename(imagename))[0] + ','
        if len(imageid2annos[imageid]) == 0:
            continue
        scores = []
        for anno in imageid2annos[imageid]:
            scores.append(anno['score'])
        max_score = max(scores)
        if max(scores) < SCORE_THR:
            keep_one = True
        else:
            keep_one = False

        for anno in imageid2annos[imageid]:
            score = anno['score']
            if score < SCORE_THR and not keep_one:
                continue
            elif score < SCORE_THR and keep_one and score < max_score:
                continue
            image_id = anno['image_id']
            assert image_id == imageid
            bbox = anno['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x_center = xmin + w / 2
            y_center = ymin + h / 2
            x_center /= 256
            y_center /= 256
            w /= 256
            h /= 256

            all_anno += str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + ';'

        all_anno = all_anno[:-1]
        all_anno += "\n"
        fp.write(all_anno)

        all_img.remove(os.path.splitext(os.path.basename(imagename))[0])
    print(len(all_img))
    for im in all_img:
        all_anno = im + ',' + '\n'
        fp.write(all_anno)
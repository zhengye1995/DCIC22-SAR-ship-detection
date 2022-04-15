# -*- coding: utf-8 -*-

import numpy as np
import os
import json
from glob import glob
from mmcv.ops import nms
from tqdm import tqdm
import torch
from mmdet.core import multiclass_nms
from ensemble_boxes import weighted_boxes_fusion

def get_all_img_in_json(json_path):

    with open(json_path, 'r') as load_f:
        json_data = json.load(load_f)

    all_images = []
    for i, box in enumerate(json_data):
        name = box['name']
        if name not in all_images:
            all_images.append(name)
    return all_images

def ensemble(submit_path, save_path, max_class_id, th_nmsiou, th_score, weights=None):
    submit_paths = glob(submit_path+'/*.json')
    # img_list = get_all_img_in_json(submit_paths[0])
    id2annos = {}
    final_result = []
    for json_file in submit_paths:
        with open(json_file, 'r') as f:
            result = json.load(f)
        for box in result:
            img_id = box['image_id']
            if img_id not in id2annos:
                id2annos[img_id] = []
            if weights is not None:
                box['score'] *= weights[os.path.basename(json_file)]
            id2annos[img_id].append(box)

    nms_cfg = dict(type='nms', iou_thr=th_nmsiou)  # nms_cfg = dict(type='soft_nms', iou_thr=th_nms, min_score=0.01)
    for id, annos in id2annos.items():
        multi_scores = []
        boxes = []
        for anno in annos:
            box = anno['bbox']
            xmin = box[0]
            ymin = box[1]
            weight = box[2]
            height = box[3]
            confidence = anno["score"]
            label_class = anno["category_id"]
            scores = []
            for _ in range(max_class_id+1):
                scores.append(0)
            scores[int(label_class)] = confidence
            multi_scores.append(scores)
            boxes.append([xmin, ymin, xmin + weight, ymin + height])

        boxes = torch.from_numpy(np.array(boxes, dtype='float32'))
        multi_scores = torch.from_numpy(np.array(multi_scores, dtype='float32'))
        if boxes.shape[0] == 0:
            continue
        det_bboxes, det_labels = multiclass_nms(boxes, multi_scores, th_score, nms_cfg, 200)
        det_bboxes[:, 2] = det_bboxes[:, 2] - det_bboxes[:, 0]
        det_bboxes[:, 3] = det_bboxes[:, 3] - det_bboxes[:, 1]
        for i in range(det_bboxes.shape[0]):
            x, y, w, h, score = det_bboxes[i]
            x = round(float(x), 4)
            y = round(float(y), 4)
            w = round(float(w), 4)
            h = round(float(h), 4)
            score = float(score)
            label = int(det_labels[i])
            final_result.append({'image_id': id, "bbox":[x, y, w, h], "score":score, "category_id":label+1})
    with open(save_path, 'w') as fp:
        json.dump(final_result, fp, indent=1)


def ensemble_wbf(submit_path, save_path, th_nmsiou, th_score, weights=None):

    submit_paths = [os.path.join(submit_path, 'swin_mosaic_submit.bbox.json'),  
                    os.path.join(submit_path, 'convnext_submit.bbox.json'),  
                    os.path.join(submit_path, 'convnext_mosaic_submit.bbox.json'),  
                    ]
    model_nums = len(submit_paths)
    final_result = []
    id2annos = [{} for _ in range(model_nums)]
    model_weights = [1 for _ in range(model_nums)]
    for i, json_file in enumerate(submit_paths):
        print(json_file)
        if weights is not None:
            model_weights[i] = weights[os.path.basename(json_file)]
        with open(json_file, 'r') as f:
            result = json.load(f)
        for box in result:
            img_id = box['image_id']
            if img_id not in id2annos[i]:
                id2annos[i][img_id] = []
            id2annos[i][img_id].append(box)

    iou_thr = th_nmsiou
    skip_box_thr = th_score
    for id, _ in id2annos[0].items():
        scores_list = [[] for _ in range(model_nums)]
        boxes_list = [[] for _ in range(model_nums)]
        labels_list = [[] for _ in range(model_nums)]

        for j in range(model_nums):
            if id in id2annos[j]:
                for anno in id2annos[j][id]:
                    box = anno['bbox']
                    xmin = box[0]
                    ymin = box[1]
                    width = box[2]
                    height = box[3]
                    xmax = xmin + width
                    ymax = ymin + height

                    xmax = xmax / 256
                    xmin = xmin / 256
                    ymin = ymin / 256
                    ymax = ymax / 256
                    confidence = anno["score"]
                    label_class = anno["category_id"]
                    scores_list[j].append(confidence)
                    boxes_list[j].append([xmin, ymin, xmax, ymax])
                    labels_list[j].append(label_class)

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=model_weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='max')
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            score = float(scores[i])
            label = int(labels[i])
            x1 = round(float(x1*256), 4)
            y1 = round(float(y1*256), 4)
            x2 = round(float(x2*256), 4)
            y2 = round(float(y2*256), 4)
            final_result.append({'image_id': id, "bbox": [x1, y1, x2-x1, y2-y1], "score": score, "category_id": label})
    with open(save_path, 'w') as fp:
        json.dump(final_result, fp, indent=1)


if __name__ == '__main__':
    submit_path = '/data/user_data/results_testa_submit'
    save_path = '/data/user_data/results_testa_submit/wbf.json'

    weights = None
    ensemble_wbf(submit_path, save_path, th_nmsiou=0.4, th_score=0.1, weights=weights)






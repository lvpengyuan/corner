#!/usr/bin/python  
# -*- coding:utf-8-*-  

import cv2
from PIL import Image, ImageDraw
import numpy as np
import math
import torch


class AnnotationTransform(object):
    def __init__(self,  keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        bboxes = target['boxes']
        diff = target['difficult']
        for i, box in enumerate(bboxes):
            is_difficult = diff[i] 
            if not self.keep_difficult and is_difficult:
                continue
            else:
                res += [box]  
        return res  # [[x1, y1, x2, y2, x3, y3, x4, y4], ... ]


def get_tight_rect(bb):
    ## bb: [x1, y1, x2, y2.......]
    points = []
    for i in range(len(bb)/2):
        points.append([int(bb[2*i]), int(bb[2*i+1])])
    bounding_box = cv2.minAreaRect(np.array(points))
    points = cv2.boxPoints(bounding_box)
    points = list(points)
    ps = sorted(points,key = lambda x:x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0]
        py1 = ps[0][1]
        px4 = ps[1][0]
        py4 = ps[1][1]
    else:
        px1 = ps[1][0]
        py1 = ps[1][1]
        px4 = ps[0][0]
        py4 = ps[0][1]
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0]
        py2 = ps[2][1]
        px3 = ps[3][0]
        py3 = ps[3][1]
    else:
        px2 = ps[3][0]
        py2 = ps[3][1]
        px3 = ps[2][0]
        py3 = ps[2][1]

    return [px1, py1, px2, py2, px3, py3, px4, py4]


def get_boxes(gt_paths):
    samples = []
    for i, timg in enumerate(gt_paths):
        item = {}
        lines = open(gt_paths[i], 'r').readlines()
        item['boxes'] = []
        item['difficult'] = []
        for line in lines:
            parts = line.strip().split(',')
            label = parts[-1]
            if '\xef\xbb\xbf' in parts[0]:
                parts[0] = parts[0][3:]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, parts[:8]))
            ## check and norm
            box = get_tight_rect([x1, y1, x2, y2, x3, y3, x4, y4])
            item['boxes'].append(box)
            if parts[-1] != '###':
                item['difficult'].append(False)
            else:
                item['difficult'].append(True)
        samples.append(item)

    return samples



# def get_boxes(gt_paths):
#     samples = []
#     for i, timg in enumerate(gt_paths):
#         item = {}
#         lines = open(gt_paths[i], 'r').readlines()
#         item['boxes'] = []
#         item['difficult'] = []
#         for line in lines:
#             parts = line.strip().split(',')
#             x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, parts[:8]))
#             ## check and norm
#             box = get_tight_rect([x1, y1, x2, y2, x3, y3, x4, y4])
#             item['boxes'].append(box)
#             if parts[-1] == '###':
#                 item['difficult'].append(True)
#             else:
#                 item['difficult'].append(False)
#         samples.append(item)
#     return samples


def generate_gt(boxes, dim=(512, 512)):
    top_left = []
    top_right = []
    bottom_right = []
    bottom_left = []
    seg = np.zeros((4, dim[0], dim[1]))

    if boxes.size > 0:
        top_left_mask = Image.new('L', (dim[1], dim[0]))
        top_left_draw = ImageDraw.Draw(top_left_mask)
        top_right_mask = Image.new('L', (dim[1], dim[0]))
        top_right_draw = ImageDraw.Draw(top_right_mask)
        bottom_right_mask = Image.new('L', (dim[1], dim[0]))
        bottom_right_draw = ImageDraw.Draw(bottom_right_mask)
        bottom_left_mask = Image.new('L', (dim[1], dim[0]))
        bottom_left_draw = ImageDraw.Draw(bottom_left_mask)
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, x3, y3, x4, y4 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][4], boxes[i][5], boxes[i][6], boxes[i][7]
            ## get box
            side1 = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
            side2 = math.sqrt(math.pow(x3 - x2, 2) + math.pow(y3 - y2, 2))
            side3 = math.sqrt(math.pow(x4 - x3, 2) + math.pow(y4 - y3, 2))
            side4 = math.sqrt(math.pow(x1 - x4, 2) + math.pow(y1 - y4, 2))
            h = min(side1 + side3, side2 + side4)/2.0
            if h*dim[0] >=6:
                theta = math.atan2(y2 - y1, x2 - x1)
                top_left.append(np.array([x1 - h/2, y1 - h/2, x1 + h/2, y1 + h/2, theta, 1]))
                top_right.append(np.array([x2 - h/2, y2 - h/2, x2 + h/2, y2 + h/2, theta, 1]))
                bottom_right.append(np.array([x3 - h/2, y3 - h/2, x3 + h/2, y3 + h/2, theta, 1]))
                bottom_left.append(np.array([x4 - h/2, y4 - h/2, x4 + h/2, y4 + h/2, theta, 1]))
                ## get seg mask
                c1_x, c2_x, c3_x, c4_x, c_x = (x1 + x2)/2.0, (x2 + x3)/2.0, (x3 + x4)/2.0, (x4 + x1)/2.0, (x1 + x2 + x3 + x4)/4.0 
                c1_y, c2_y, c3_y, c4_y, c_y = (y1 + y2)/2.0, (y2 + y3)/2.0, (y3 + y4)/2.0, (y4 + y1)/2.0, (y1 + y2 + y3 + y4)/4.0
                top_left_draw.polygon([x1*dim[1], y1*dim[0], c1_x*dim[1], c1_y*dim[0], c_x*dim[1], c_y*dim[0], c4_x*dim[1], c4_y*dim[0]], fill = 1)
                top_right_draw.polygon([c1_x*dim[1], c1_y*dim[0], x2*dim[1], y2*dim[0], c2_x*dim[1], c2_y*dim[0], c_x*dim[1], c_y*dim[0]], fill = 1)
                bottom_right_draw.polygon([c_x*dim[1], c_y*dim[0], c2_x*dim[1], c2_y*dim[0], x3*dim[1], y3*dim[0], c3_x*dim[1], c3_y*dim[0]], fill = 1)
                bottom_left_draw.polygon([c4_x*dim[1], c4_y*dim[0], c_x*dim[1], c_y*dim[0], c3_x*dim[1], c3_y*dim[0], x4*dim[1], y4*dim[0]], fill = 1)
        seg[0] = top_left_mask
        seg[1] = top_right_mask
        seg[2] = bottom_right_mask
        seg[3] = bottom_left_mask
        if len(top_left) == 0:
            top_left.append(np.array([-1, -1, -1, -1, 0, 0]))
            top_right.append(np.array([-1, -1, -1, -1, 0, 0]))
            bottom_right.append(np.array([-1, -1, -1, -1, 0, 0]))
            bottom_left.append(np.array([-1, -1, -1, -1, 0, 0]))

    else:
        top_left.append(np.array([-1, -1, -1, -1, 0, 0]))
        top_right.append(np.array([-1, -1, -1, -1, 0, 0]))
        bottom_right.append(np.array([-1, -1, -1, -1, 0, 0]))
        bottom_left.append(np.array([-1, -1, -1, -1, 0, 0]))

    top_left = torch.FloatTensor(np.array(top_left))
    top_right = torch.FloatTensor(np.array(top_right))
    bottom_right = torch.FloatTensor(np.array(bottom_right))
    bottom_left = torch.FloatTensor(np.array(bottom_left))

    seg = torch.from_numpy(seg).float()
    seg = seg.permute(1, 2, 0).contiguous()
    return [top_left, top_right, bottom_right, bottom_left], seg

def detection_collate(batch):
    boxes = []
    imgs = []
    segs = []
    for sample in batch:
        imgs.append(sample[0])
        boxes.append(sample[1])
        segs.append(sample[2])
    return torch.stack(imgs, 0), boxes, torch.stack(segs, 0)
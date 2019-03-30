#!/usr/bin/python  
#-*- coding:utf-8 -*-


import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import math
from utils import get_boxes, generate_gt


class MLTDetection(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None, dim=(512, 512)):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.dim = dim
        
        if self.split == 'train':
            all_items = os.listdir(self.root + '/training/')
            imgs = [img for img in all_items if img.strip().split('.')[-1] == 'jpg']
            gts = ['gt_' + x.strip().split('.')[0] + '.txt' for x in imgs]
            image_paths = [self.root + '/training/' + x for x in imgs]
            gt_paths = [self.root + '/training/' + x for x in gts]
            self.image_paths = image_paths
            self.targets = get_boxes(gt_paths)
        elif self.split == 'val':
            all_items = os.listdir(self.root + '/validation/')
            imgs = [img for img in all_items if img.strip().split('.')[-1] == 'jpg']
            gts = ['gt_' + x.strip().slit('.')[0] + '.txt' for x in imgs]
            image_paths = [self.root + '/validation/' + x for x in imgs]
            gt_paths = [self.root + '/validation/' + x for x in gts]
            self.image_paths = image_paths
            self.targets = get_boxes(gt_paths)
        else:
            all_items  = os.listdir(self.root + '/test/')
            image_paths = [self.root + '/test/' + x for x in all_items]
            self.image_paths = image_paths
            self.targets = []

    def __getitem__(self, index):
        return  self.pull_item(index)
    
    def __len__(self):
        return len(self.image_paths)

    

    

    def pull_item(self, index, model='debug'):
        img_path = self.image_paths[index]
        #target = self.targets[index]
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        if self.split != 'train':
            img = cv2.resize(img, (self.dim[1], self.dim[0])).astype(np.float64)
            img -= np.array([104.00698793, 116.66876762, 122.67891434])  ## mean -bgr
            img = img[:, :, (2, 1, 0)] ## rgb
            return torch.from_numpy(img).permute(2, 0, 1).float(), img_path, height, width
        else:
            target = self.targets[index]
            ## get rotate rect  [x1, y1, x2, y2, x3, y3, x4, y4, diff]
            target = self.target_transform(target)
            assert(self.transform)

            target = np.array(target)
            img, boxes, labels = self.transform(img, target, None)
            img = img[:, :, (2, 1, 0)]
            target, seg = generate_gt(boxes)
            return torch.from_numpy(img).permute(2, 0, 1), target, seg
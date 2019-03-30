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

class ICDARDetection(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None,
                 dataset_name='13&&15', dim=(512, 512)):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.dim = dim
        
        assert(self.name == "13&&15" or self.name == '13' or self.name == '15')
        if self.split == 'train':
            assert(self.target_transform)
            ## get ic13 list
            ic13_list_path = self.root + '/icdar2013/train_list.txt'
            ic13_samples = open(ic13_list_path, 'r').readlines()
            ic13_gt_paths = [self.root + '/icdar2013/train_gts/gt_' + timg.strip().split('.')[0] + '.txt' for timg in ic13_samples]
            ic13_img_paths = [self.root + '/icdar2013/train_images/' + timg.strip() for timg in ic13_samples]
            ## get ic15 list
            ic15_list_path = self.root + 'icdar2015/train_list.txt'
            ic15_samples = open(ic15_list_path, 'r').readlines()
            ic15_gt_paths = [self.root + '/icdar2015/train_gts/gt_' + timg.strip().split('.')[0] + '.txt' for timg in ic15_samples]
            ic15_img_paths = [self.root + '/icdar2015/train_images/' + timg.strip() for timg in ic15_samples]

            if self.name == '13&&15':
                image_paths = ic13_img_paths + ic15_img_paths
                gt_paths = ic13_gt_paths + ic15_gt_paths
            elif self.name == '13':
                image_paths = ic13_img_paths
                gt_paths = ic13_gt_paths
            else:
                image_paths = ic15_img_paths
                gt_paths = ic15_gt_paths
        else:

            ic13_list_path = self.root + '/icdar2013/test_list.txt'
            ic13_samples = open(ic13_list_path, 'r').readlines()
            ic13_gt_paths = [self.root + '/icdar2013/test_gts/' + timg.strip() + '.txt' for timg in ic13_samples]
            ic13_img_paths = [self.root + '/icdar2013/test_images/' + timg.strip() for timg in ic13_samples]
            ## get ic15 list
            ic15_list_path = self.root + 'icdar2015/test_list.txt'
            ic15_samples = open(ic15_list_path, 'r').readlines()
            ic15_gt_paths = [self.root + '/icdar2015/test_gts/' + timg.strip() + '.txt' for timg in ic15_samples]
            ic15_img_paths = [self.root + '/icdar2015/test_images/' + timg.strip() for timg in ic15_samples]
            assert(self.name == '13' or self.name == '15')
            if self.name == '13':
                image_paths = ic13_img_paths
                # gt_paths = ic13_gt_paths
                gt_paths = []

            else:
                image_paths = ic15_img_paths
                gt_paths = ic15_gt_paths

        self.image_paths = image_paths
        self.targets = get_boxes(gt_paths)

    def __getitem__(self, index):
        return  self.pull_item(index)
    
    def __len__(self):
        return len(self.image_paths)

    def pull_item(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        if self.split != 'train':
            img = cv2.resize(img, (self.dim[1], self.dim[0])).astype(np.float64)
            img -= np.array([104.00698793, 116.66876762, 122.67891434])  ## mean -bgr
            img = img[:, :, (2, 1, 0)] ## rgb
            return torch.from_numpy(img).permute(2, 0, 1).float(), img_path, height, width
        else:
            ## get rotate rect  [x1, y1, x2, y2, x3, y3, x4, y4, diff]
            target = self.targets[index]
            target = self.target_transform(target)
            assert(self.transform)

            target = np.array(target)
            img, boxes, labels = self.transform(img, target, None)
            img = img[:, :, (2, 1, 0)]
            target, seg = generate_gt(boxes)
            return torch.from_numpy(img).permute(2, 0, 1), target, seg

   


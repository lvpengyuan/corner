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

class TD500Detection(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None,
                 aug=True, dim=(512, 512)):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug
        self.dim = dim
        
        if self.split == 'train':
            assert(self.target_transform)
            ## get td500 list
            td500_list_path = self.root + '/TD500/train_list.txt'
            td500_samples = open(td500_list_path, 'r').readlines()
            td500_gt_paths = [self.root + '/TD500/train_gts/' + timg.strip().split('.')[0] + '.gt' for timg in td500_samples]
            td500_img_paths = [self.root + '/TD500/train_images/' + timg.strip() for timg in td500_samples]
            ## get tr400 list
            tr400_list_path = self.root + '/TR400/train_list.txt'
            tr400_samples = open(tr400_list_path, 'r').readlines()
            tr400_gt_paths = [self.root + '/TR400/train_gts/' + timg.strip().split('.')[0] + '.gt' for timg in tr400_samples]
            tr400_img_paths = [self.root + '/TR400/train_images/' + timg.strip() for timg in tr400_samples]

            if self.aug== True:
                image_paths = td500_img_paths + tr400_img_paths
                gt_paths = td500_gt_paths + tr400_gt_paths
            else:
                image_paths = tr500_img_paths
                gt_paths = tr400_gt_paths
        else:

            td500_list_path = self.root + '/TD500/test_list.txt'
            td500_samples = open(td500_list_path, 'r').readlines()
            td500_gt_paths = [self.root + '/TD500/test_gts/' + timg.strip().split('.')[0] + '.gt' for timg in td500_samples]
            td500_img_paths = [self.root + '/TD500/test_images/' + timg.strip() for timg in td500_samples]  
            image_paths = td500_img_paths
            gt_paths = td500_gt_paths

        self.image_paths = image_paths
        self.targets = get_boxes(gt_paths)

    def __getitem__(self, index):
        return  self.pull_item(index)
    
    def __len__(self):
        return len(self.image_paths)



    def pull_item(self, index, model='debug'):
        img_path = self.image_paths[index]
        target = self.targets[index]
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        if self.split != 'train':
            img = cv2.resize(img, (self.dim[1], self.dim[0])).astype(np.float64)
            img -= np.array([104.00698793, 116.66876762, 122.67891434])  ## mean -bgr
            img = img[:, :, (2, 1, 0)] ## rgb
            return torch.from_numpy(img).permute(2, 0, 1).float(), img_path, height, width
        else:
            
            ## get rotate rect  [x1, y1, x2, y2, x3, y3, x4, y4, diff]
            target = self.target_transform(target)
            assert(self.transform)

            target = np.array(target)
            img, boxes, labels = self.transform(img, target, None)
            img = img[:, :, (2, 1, 0)]
            
            target, seg = generate_gt(boxes)
            return torch.from_numpy(img).permute(2, 0, 1), target, seg

   
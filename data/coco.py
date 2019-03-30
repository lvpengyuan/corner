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

## Data Loader of COCO-Text
class COCODetection(data.Dataset):
    def __init__(self, root, split,  dim=(512, 512)):
        self.root = root
        self.split = split
        self.dim = dim

        coco_test_list_path = self.root + '/test.list'
        coco_val_list_path = self.root + '/val.list'
        coco_tests = open(coco_test_list_path, 'r').readlines()
        coco_vals = open(coco_val_list_path, 'r').readlines()
        coco_test_img_path = [self.root + '/coco_test/' + timg.strip() for timg in coco_tests]
        coco_val_img_path = [self.root + '/coco_val/' + timg.strip() for timg in coco_vals]

        
        if self.split == 'test':
            self.image_paths = coco_test_img_path
        elif self.split == 'val':
            self.image_paths = coco_val_img_path
        elif self.split == 'val&&test':
            self.image_paths = coco_test_img_path + coco_val_img_path
        else:
            print('error, split shoulb be val, test, or val&&test')
            exit()
        

    def __getitem__(self, index):
        return  self.pull_item(index)
    
    def __len__(self):
        return len(self.image_paths)

    def pull_item(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        
        img = cv2.resize(img, (self.dim[1], self.dim[0])).astype(np.float64)
        img -= np.array([104.00698793, 116.66876762, 122.67891434])  ## mean -bgr
        img = img[:, :, (2, 1, 0)] ## rgb
        return torch.from_numpy(img).permute(2, 0, 1).float(), img_path, height, width
        

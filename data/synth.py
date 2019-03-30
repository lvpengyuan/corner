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

class SynthDetection(data.Dataset):
    
    def __init__(self, root, transform=None, target_transform=None,dim=(512, 512)):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dim = dim
        assert(self.target_transform)

        synth_samples = open(self.root + '/train_list.txt', 'r').readlines()
        synth_gt_paths = [self.root + '/SynthText_GT/' + x.strip() + '.txt' for x in synth_samples]
        synth_img_paths = [self.root + '/SynthText/' + x.strip() for x in synth_samples]
        
        self.image_paths = synth_img_paths
        self.targets = get_boxes(synth_gt_paths)

    def __getitem__(self, index):
        return  self.pull_item(index)
    
    def __len__(self):
        return len(self.image_paths)

    def pull_item(self, index):
        img_path = self.image_paths[index]
        target = self.targets[index]
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        ## get rotate rect  [x1, y1, x2, y2, x3, y3, x4, y4, diff]
        target = self.target_transform(target)
        assert(self.transform)

        target = np.array(target)
        img, boxes, labels = self.transform(img, target, None)
        img = img[:, :, (2, 1, 0)]
        target, seg = generate_gt(boxes, dim)
        return torch.from_numpy(img).permute(2, 0, 1), target, seg

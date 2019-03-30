#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import cv2
import numpy as np
import sys

def rbox2polygon(rbox):
  x_center=rbox[0]
  y_center=rbox[1]
  w=rbox[2]
  h=rbox[3]
  theta=rbox[4]
  # let (x_center,y_center) be the (0,0)
  x1_re=-w/2
  y1_re=-h/2
  x2_re=w/2
  y2_re=-h/2
  x3_re=w/2
  y3_re=h/2
  x4_re=-w/2
  y4_re=h/2
  x1=math.cos(theta)*x1_re-math.sin(theta)*y1_re+x_center
  y1=math.cos(theta)*y1_re+math.sin(theta)*x1_re+y_center
  x2=math.cos(theta)*x2_re-math.sin(theta)*y2_re+x_center
  y2=math.cos(theta)*y2_re+math.sin(theta)*x2_re+y_center
  x3=math.cos(theta)*x3_re-math.sin(theta)*y3_re+x_center
  y3=math.cos(theta)*y3_re+math.sin(theta)*x3_re+y_center
  x4=math.cos(theta)*x4_re-math.sin(theta)*y4_re+x_center
  y4=math.cos(theta)*y4_re+math.sin(theta)*x4_re+y_center
  polygon=[x1,y1,x2,y2,x3,y3,x4,y4]
  return polygon

def polygon2rbox(polygon):
    x_center=(polygon[0]+polygon[2]+polygon[4]+polygon[6])/4.0
    y_center=(polygon[1]+polygon[3]+polygon[5]+polygon[7])/4.0
    w=math.sqrt((polygon[0]-polygon[2])*(polygon[0]-polygon[2])+(polygon[1]-polygon[3])*(polygon[1]-polygon[3]))
    h=math.sqrt((polygon[0]-polygon[6])*(polygon[0]-polygon[6])+(polygon[1]-polygon[7])*(polygon[1]-polygon[7]))
    xx=(polygon[2]+polygon[4])/2.0
    yy=(polygon[3]+polygon[5])/2.0
    tan_theta=(yy-y_center+1e-6)/(xx-x_center+1e-6)
    theta=math.atan(tan_theta)
    rbox=[x_center,y_center,w,h,theta]
    return rbox


def calculate_iou(bbox1, bbox2):
    rbox1=polygon2rbox(bbox1)
    # rotate_rbox1=[rbox1[0],rbox1[1],rbox1[3],rbox1[2]]
    # rbox2=polygon2rbox(bbox2,expand=True)
    rbox2=polygon2rbox(bbox2)
    # if abs(rbox1[4]-rbox2[4])<=math.pi/8 or abs(abs(rbox1[4]-rbox2[4])-math.pi/4)<=math.pi/8:
    if abs(abs(rbox1[4])-abs(rbox2[4]))<=math.pi/8:
        # calculate iou
        xmin1=rbox1[0]-rbox1[2]/2.0
        ymin1=rbox1[1]-rbox1[3]/2.0
        xmax1=rbox1[0]+rbox1[2]/2.0
        ymax1=rbox1[1]+rbox1[3]/2.0

        # rotate_xmin1=rotate_rbox1[0]-rotate_rbox1[2]/2.0
        # rotate_ymin1=rotate_rbox1[1]-rotate_rbox1[3]/2.0
        # rotate_xmax1=rotate_rbox1[0]+rotate_rbox1[2]/2.0
        # rotate_ymax1=rotate_rbox1[1]+rotate_rbox1[3]/2.0

        xmin2=rbox2[0]-rbox2[2]/2.0
        ymin2=rbox2[1]-rbox2[3]/2.0
        xmax2=rbox2[0]+rbox2[2]/2.0
        ymax2=rbox2[1]+rbox2[3]/2.0
        intersection = max(0, min(xmax1, xmax2) - max(xmin1, xmin2)) * max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
        # rotate_intersection = max(0, min(rotate_xmax1, xmax2) - max(rotate_xmin1, xmin2)) * max(0, min(rotate_ymax1, ymax2) - max(rotate_ymin1, ymin2))
        area1=(xmax1-xmin1)*(ymax1-ymin1)
        # rotate_area1=(rotate_xmax1-rotate_xmin1)*(rotate_ymax1-rotate_ymin1)
        area2=(xmax2-xmin2)*(ymax2-ymin2)
        # rotate_iou=float(rotate_intersection)/(rotate_area1+area2-rotate_intersection)
        iou=float(intersection)/(area1+area2-intersection)
        # iou=max(iou,rotate_iou)
    else:
        iou=0
    
    return iou

if __name__ == '__main__':
    # detection_results_dir='./msra_rbox/'
    # detection_results_dir='/home/lvpengyuan/research/dssd_box.pytorch/outputs/imgs/td500/240/res/'
    detection_results_dir='/home/lvpengyuan/research/cvpr18_code/car.pytorch/outputs/imgs/td500/240/res/'

    
    gt_dir='/home/lvpengyuan/research/text/MSRA-TD500_ORI/test/'
    count=0
    tp=0
    fp=0
    tn=0
    ta=0
    for root, dirs, files in os.walk(detection_results_dir):
        for file in files:
            count=count+1
            # print(count)
            # print(file)
            result_path=detection_results_dir+file
            gt_path=gt_dir+file[4:len(file)-4]+'.gt'
            result_fid=open(result_path,'r')
            gt_fid=open(gt_path,'r')
            result_boxes=[]
            gt_boxes=[]
            difficults=[]
            for line in result_fid.readlines():
                line=line.strip()
                x1=int(line.split(',')[0])
                y1=int(line.split(',')[1])
                x2=int(line.split(',')[2])
                y2=int(line.split(',')[3])
                x3=int(line.split(',')[4])
                y3=int(line.split(',')[5])
                x4=int(line.split(',')[6])
                y4=int(line.split(',')[7])
                rbox_tmp=polygon2rbox([x1,y1,x2,y2,x3,y3,x4,y4])
                # if float(rbox_tmp[2])/rbox_tmp[3]>2:
                result_boxes.append([x1,y1,x2,y2,x3,y3,x4,y4])
                # result_boxes.append([x1,y1,x2,y2,x3,y3,x4,y4])

            for line in gt_fid.readlines():
                line=line.strip()
                difficult=int(line.split(' ')[1])
                x1=float(line.split(' ')[2])
                y1=float(line.split(' ')[3])
                w=float(line.split(' ')[4])
                h=float(line.split(' ')[5])
                x_center=x1+w/2.0
                y_center=y1+h/2.0
                theta=float(line.split(' ')[6])
                word_polygon=rbox2polygon([x_center,y_center,w,h,theta])
                box=list(np.int0(word_polygon))
                gt_boxes.append(box)
                difficults.append(difficult)

            ta=ta+len(result_boxes)
            for i in range(len(gt_boxes)):
                gt_box=gt_boxes[i]
                difficult=difficults[i]
                flag=0
                for result in result_boxes:
                    iou=calculate_iou(gt_box, result)
                    if iou>=0.5:
                        flag=1
                        tp=tp+1
                        break

                if flag==0 and difficult==0:
                    fp=fp+1

    recall=float(tp)/(tp+fp)
    precision=float(tp)/ta
    f_measure=2*(precision*recall)/(precision+recall)
    # recall=round(float(tp)/(tp+fp),2)
    # precision=round(float(tp)/ta,2)
    # f_measure=2*(precision*recall)/(precision+recall)
    print('recall: '+str(recall))
    print('precision: '+str(precision))
    print('f_measure: '+str(f_measure))



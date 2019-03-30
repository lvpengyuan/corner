import os
import torch
import torch.nn as nn
import argparse
import torch.utils.data as data
from torch.autograd import Variable
from data import  train_cfg, cfg_768x1280, cfg_512x512, cfg_768x768, cfg_1280x1280, AnnotationTransform, ICDARDetection, detection_collate, TD500Detection, COCODetection
from utils.augmentations_poly import SSDAugmentation
from utils.logger import setup_logger
from dssd import build_dssd
import numpy as np
import time
import logging
from PIL import Image, ImageDraw
import math
import cv2
from shapely.geometry import box, Polygon
from rpsroi_pooling.modules.rpsroi_pool import RPSRoIPool

def edge_len(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)*(x2-x1) + (y2 - y1)*(y2-y1))


def ploy_nms(boxes, thresh):
    ploys = [Polygon([[x[0], x[1]], [x[2], x[3]], [x[4], x[5]], [x[6], x[7]]]) for x in boxes]
    scores = [x[-1] for x in boxes]
    areas = [x.area for x in ploys]

    order = np.array(scores).argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = []
        for j in order[1:]:
            inter = ploys[i].intersection(ploys[j]).area
            ious.append(inter/(ploys[i].area + ploys[j].area - inter))
        inds = np.where(np.array(ious) <= thresh)[0]
        order = order[inds + 1]
    return keep

def get_score_rpsroi(bboxes, seg_cuda, rpsroi_pool):
    if len(bboxes) > 0:
        sample_index = torch.zeros(len(bboxes)).view(-1, 1).cuda()
        bboxes = torch.from_numpy(np.array(bboxes)).float().cuda()
        rois = Variable(torch.cat((sample_index, bboxes), 1))
        seg_cuda  = seg_cuda.data
        seg_cuda = torch.index_select(seg_cuda, 1, torch.LongTensor([0, 1, 3, 2]).cuda())
        seg_cuda = Variable(seg_cuda)
        rps_score = rpsroi_pool.forward(seg_cuda, rois)
        return rps_score.data.cpu().view(-1, 4).mean(1).numpy()
    else:
        return np.array([-1])

def get_score(bbox, seg_pred):
    ## check
    seg_pred = seg_pred.numpy()
    mask = np.zeros(seg_pred.shape)
    c1_x, c2_x, c3_x, c4_x, c_x = (bbox[0] + bbox[2])/2.0, (bbox[2] + bbox[4])/2.0, (bbox[4] + bbox[6])/2.0, (bbox[6] + bbox[0])/2.0, (bbox[0] + bbox[2] + bbox[4] + bbox[6])/4.0
    c1_y, c2_y, c3_y, c4_y, c_y = (bbox[1] + bbox[3])/2.0, (bbox[3] + bbox[5])/2.0, (bbox[5] + bbox[7])/2.0, (bbox[7] + bbox[1])/2.0, (bbox[1] + bbox[3] + bbox[5] + bbox[7])/4.0
    cv2.fillConvexPoly(mask[0], np.array([[bbox[0], bbox[1]*1.0], [c1_x, c1_y], [c_x, c_y], [c4_x, c4_y]]).astype(np.int32), 1)
    cv2.fillConvexPoly(mask[1], np.array([[c1_x, c1_y], [bbox[2]*1.0, bbox[3]*1.0], [c2_x, c2_y], [c_x, c_y]]).astype(np.int32), 1)
    cv2.fillConvexPoly(mask[2], np.array([[c_x, c_y], [c2_x, c2_y], [bbox[4]*1.0, bbox[5]*1.0], [c3_x, c3_y]]).astype(np.int32), 1)
    cv2.fillConvexPoly(mask[3], np.array([[c4_x, c4_y], [c_x, c_y], [c3_x, c3_y], [bbox[6]*1.0, bbox[7]*1.0]]).astype(np.int32), 1)
    score = 0
    for i in range(4):
        score += (mask[i]*seg_pred[i]).sum()/(mask[i].sum())
    score = score/4.0/255.0
    return score

def get_boxes(top_left_points, top_right_points, bottom_right_points, bottom_left_points, seg_pred, seg_cuda, rpsroi_pool, thre):
    random_box = []
    candidate_box = []

    # top_line
    for top_left_point in top_left_points:
        for top_right_point in top_right_points:
            if top_left_point[0] < top_right_point[0] and top_left_point[2] > 5 and top_right_point[2] > 5 and max(top_left_point[2], top_right_point[2])/min(top_left_point[2], top_right_point[2]) < 1.5:
            #if top_left_point[0] < top_right_point[0]:
                side = (top_left_point[2] + top_right_point[2])/2.0
                theta = math.atan2(top_right_point[1] - top_left_point[1], top_right_point[0] - top_left_point[0]) + math.pi/2
                x3 = top_right_point[0] + math.cos(theta)*side
                y3 = top_right_point[1] + math.sin(theta)*side
                x4 = top_left_point[0] + math.cos(theta)*side
                y4 = top_left_point[1] + math.sin(theta)*side
                if edge_len(top_left_point[0], top_left_point[1], top_right_point[0], top_right_point[1]) > 5 and edge_len(top_right_point[0], top_right_point[1], x3, y3) > 5 and edge_len(x3, y3, x4, y4) > 5 and edge_len(x4, y4, top_left_point[0], top_left_point[1]) > 5:
                    random_box.append([top_left_point[0], top_left_point[1], top_right_point[0], top_right_point[1], x3, y3, x4, y4])
    ## bottom_line
    for bottom_left_point in bottom_left_points:
        for bottom_right_point in bottom_right_points:
            if bottom_left_point[0] < bottom_right_point[0] and bottom_left_point[2] > 5 and bottom_right_point[2] > 5 and max(bottom_left_point[2], bottom_right_point[2])/min(bottom_left_point[2], bottom_right_point[2]) < 1.5:
            #if bottom_left_point[0] < bottom_right_point[0]:
                side = (bottom_left_point[2] + bottom_right_point[2])/2.0
                theta = math.atan2(bottom_right_point[1] - bottom_left_point[1], bottom_right_point[0] - bottom_left_point[0]) - math.pi/2
                x2 = bottom_right_point[0] + math.cos(theta)*side
                y2 = bottom_right_point[1] + math.sin(theta)*side
                x1 = bottom_left_point[0] + math.cos(theta)*side
                y1 = bottom_left_point[1] + math.sin(theta)*side
                if edge_len(x1, y1, x2, y2) > 5 and edge_len(x2, y2, bottom_right_point[0], bottom_right_point[1]) > 5 and edge_len(bottom_right_point[0], bottom_right_point[1], bottom_left_point[0], bottom_left_point[1]) > 5 and edge_len(bottom_left_point[0], bottom_left_point[1], x1, y1) > 5: 
                    random_box.append([x1, y1, x2, y2, bottom_right_point[0], bottom_right_point[1], bottom_left_point[0], bottom_left_point[1]])
    

    ## left_line
    for top_left_point in top_left_points:
        for bottom_left_point in bottom_left_points:
            if top_left_point[1] < bottom_left_point[1] and top_left_point[2] > 5 and bottom_left_point[2] > 5 and max(top_left_point[2], bottom_left_point[2])/min(top_left_point[2], bottom_left_point[2]) < 1.5:
                side = (top_left_point[2] + bottom_left_point[2])/2.0
                theta = math.atan2(bottom_left_point[1] - top_left_point[1], bottom_left_point[0] - top_left_point[0]) - math.pi/2
                x3 = bottom_left_point[0] + math.cos(theta)*side
                y3 = bottom_left_point[1] + math.sin(theta)*side
                x2 = top_left_point[0] + math.cos(theta)*side
                y2 = top_left_point[1] + math.sin(theta)*side
                if edge_len(top_left_point[0], top_left_point[1], bottom_left_point[0], bottom_left_point[1]) > 5 and edge_len(bottom_left_point[0], bottom_left_point[1], x3, y3) > 5 and edge_len(x3, y3, x2, y2) > 5 and edge_len(x2, y2, top_left_point[0], top_left_point[1]) > 5:
                    random_box.append([top_left_point[0], top_left_point[1], x2, y2, x3, y3, bottom_left_point[0], bottom_left_point[1]])
    ## right_line
    for top_right_point in top_right_points:
        for bottom_right_point in bottom_right_points:
            if top_right_point[1] < bottom_right_point[1] and top_right_point[2] > 5 and bottom_right_point[2] > 5 and max(top_right_point[2], bottom_right_point[2])/min(top_right_point[2], bottom_right_point[2]) < 1.5:
            #if top_right_point[0] < bottom_right_point[0]:
                side = (top_right_point[2] + bottom_right_point[2])/2.0
                theta = math.atan2(bottom_right_point[1] - top_right_point[1], bottom_right_point[0] - top_right_point[0]) + math.pi/2
                x4 = bottom_right_point[0] + math.cos(theta)*side
                y4 = bottom_right_point[1] + math.sin(theta)*side
                x1 = top_right_point[0] + math.cos(theta)*side
                y1 = top_right_point[1] + math.sin(theta)*side
                if edge_len(x1, y1, x4, y4) > 5 and edge_len(x4, y4, bottom_right_point[0], bottom_right_point[1]) > 5 and edge_len(bottom_right_point[0], bottom_right_point[1], top_right_point[0], top_right_point[1]) > 5 and edge_len(top_right_point[0], top_right_point[1], x1, y1) > 5: 
                    random_box.append([x1, y1, top_right_point[0], top_right_point[1],  bottom_right_point[0], bottom_right_point[1], x4, y4])
    


    scores = get_score_rpsroi(random_box, seg_cuda, rpsroi_pool)
    for i in range(len(random_box)):
        if scores[i] > thre:
            candidate_box.append(random_box[i] + [scores[i]])

    return candidate_box

def vis(imgs, boxes, h, w):
    img = imgs[0].data.cpu().numpy().transpose(1,2,0) + np.array([122.67891434, 116.66876762, 104.00698793])
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img_draw = ImageDraw.Draw(img)
    boxes = boxes.data.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2, label = box[1]*w, box[2]*h, box[3]*w, box[4]*h, box[5]
        if label == 0:
            # img_draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255))
            img_draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill=(255, 255, 255))
        elif label == 1:
            # img_draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0))
            img_draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill=(255, 0, 0))
        elif label == 2:
            # img_draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0))
            img_draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill=(0, 255, 0))
        else:
            # img_draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255))
            img_draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill=(0, 0, 255))
    return img


def vis_seg(img, segs, name, dim):
    w, h = img.size
    seg = segs.contiguous().view(-1, h, w, 4).permute(0,3,1,2).data.cpu()[0]*255
    top_left_mask = Image.fromarray(seg[0].numpy().astype(np.uint8), 'L').convert('RGB')
    top_right_mask = Image.fromarray(seg[2].numpy().astype(np.uint8), 'L').convert('RGB')
    bottom_right_mask = Image.fromarray(seg[4].numpy().astype(np.uint8), 'L').convert('RGB')
    bottom_left_mask = Image.fromarray(seg[6].numpy().astype(np.uint8), 'L').convert('RGB')
    top_left = Image.blend(img, top_left_mask, 0.5)
    top_right = Image.blend(img, top_right_mask, 0.5)
    bottom_right = Image.blend(img, bottom_right_mask, 0.5)
    bottom_left = Image.blend(img, bottom_left_mask, 0.5)
    top_left.save(name + '_1.jpg')
    top_right.save(name + '_2.jpg')
    bottom_right.save(name + '_3.jpg')
    bottom_left.save(name + '_4.jpg')

def show_box(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon(box[:-1], outline=(255, 0, 0))
    return img

def eval_img(out, seg_pred, seg_map, rpsroi_pool, img, save_name, seg_dir, box_dir, vis=True):
    
    img = img[0].data.cpu().numpy().transpose(1,2,0) + np.array([122.67891434, 116.66876762, 104.00698793])
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    if vis:
        img_draw = ImageDraw.Draw(img)
    boxes = out.data.cpu().numpy()
    top_left_points, top_right_points, bottom_right_points, bottom_left_points = [], [], [], []
    w, h  = img.size
    for box in boxes:
        x1, y1, x2, y2, label = box[1]*w, box[2]*h, box[3]*w, box[4]*h, box[5]
        if label == 0:
            if vis:
                img_draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill=(255, 255, 255))
            top_left_points.append([(x1 + x2)/2 , (y1 + y2)/2, x2 - x1])
        elif label == 1:
            if vis:
                img_draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill=(255, 0, 0))
            top_right_points.append([(x1 + x2)/2 , (y1 + y2)/2, x2 - x1])
        elif label == 2:
            if vis:
                img_draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill=(0, 255, 0))
            bottom_right_points.append([(x1 + x2)/2 , (y1 + y2)/2, x2 - x1])
        else:
            if vis:
                img_draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill=(0, 0, 255))
            bottom_left_points.append([(x1 + x2)/2 , (y1 + y2)/2, x2 - x1])
    
    seg = seg_pred.contiguous().view(-1, h, w, 4).permute(0,3,1,2).data.cpu()[0]*255
    if vis:
        top_left_mask = Image.fromarray(seg[0].numpy().astype(np.uint8), 'L').convert('RGB')
        top_right_mask = Image.fromarray(seg[1].numpy().astype(np.uint8), 'L').convert('RGB')
        bottom_right_mask = Image.fromarray(seg[2].numpy().astype(np.uint8), 'L').convert('RGB')
        bottom_left_mask = Image.fromarray(seg[3].numpy().astype(np.uint8), 'L').convert('RGB')
        top_left = Image.blend(img, top_left_mask, 0.5)
        top_right = Image.blend(img, top_right_mask, 0.5)
        bottom_right = Image.blend(img, bottom_right_mask, 0.5)
        bottom_left = Image.blend(img, bottom_left_mask, 0.5)
        top_left.save(seg_dir + '/' + save_name + '_1.jpg')
        top_right.save(seg_dir + '/' + save_name + '_2.jpg')
        bottom_right.save(seg_dir + '/' + save_name + '_3.jpg')
        bottom_left.save(seg_dir + '/' + save_name + '_4.jpg')

    candidate_boxes = get_boxes(top_left_points, top_right_points, bottom_right_points, bottom_left_points, seg, seg_map, rpsroi_pool, 0.60)
    keep = ploy_nms(candidate_boxes, 0.3)
    keep_box = []
    for j, item in enumerate(candidate_boxes):
        if j in keep:
            keep_box.append(item)
    if vis:
        box_img = show_box(img, keep_box)
        box_img.save(box_dir + '/' + save_name + '.jpg')
    return keep_box





def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Testing')
    parser.add_argument('--resume', dest='resume',
                        help='initialize with pretrained model weights',
                        default='./weights/ic15_90_15.pth', type=str)
    parser.add_argument('--version', dest='version', help='512x512, 768x768, 768x1280, 1280x1280', default='768x1280', type=str)
    parser.add_argument('--dataset', dest='dataset', help = 'ic15, ic13, td500, coco'
                        ,default='ic15', type=str)
    parser.add_argument('--works', dest='num_workers',
                        help='num_workers to load data',
                        default=1, type=int)
    parser.add_argument('--test_batch_size', dest='test_batch_size',
                        help='train_batch_size',
                        default=1, type=int)
    parser.add_argument('--out', dest='out',
                        help='output file dir',
                        default='./outputs_eval/ic15/', type=str)
    parser.add_argument('--log_file_dir', dest='log_file_dir',
                        help='log_file_dir',
                        default='./logs/', type=str)
    parser.add_argument('--ssd_dim', default=512, type=int, help='ssd dim')

    #parser.add_argument('--root', default='../../DataSets/text_detect/',type=str,  help='Location of data root directory')
    parser.add_argument('--ic_root', default='../data/ocr/detection/',type=str,  help='Location of data root directory')
    # parser.add_argument('--ic_root', default='/home/lvpengyuan/research/text/',type=str,  help='Location of data root directory')
    parser.add_argument('--td_root', default='/home/lpy/Datasets/TD&&TR/',type=str,  help='Location of data root directory')
    parser.add_argument('--coco_root', default='/home/lpy/Datasets/coco-text/', type=str, help='Location of data root direction')
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    ## setup logger
    if os.path.exists(args.log_file_dir) == False:
        os.mkdir(args.log_file_dir)
    log_file_path = args.log_file_dir + 'eval_' + time.strftime('%Y%m%d_%H%M%S') + '.log'
    setup_logger(log_file_path)

    if args.version == '512x512':
        cfg = cfg_512x512
    elif args.version == '768x768':
        cfg = cfg_768x768
    elif args.version == '1280x1280':
        cfg = cfg_1280x1280
    elif args.version == '768x1280':
        cfg = cfg_768x1280
    else:
        exit()


    ssd_dim = args.ssd_dim
    means = (104, 117, 123) 
    
    if args.dataset == 'ic15':
        dataset = ICDARDetection(args.ic_root, 'val',None, None, '15', dim=cfg['min_dim'])
        data_loader = data.DataLoader(dataset, args.test_batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=True)
    elif args.dataset == 'ic13':
        dataset = ICDARDetection(args.ic_root, 'val',None, None, '13', dim=cfg['min_dim'])
        data_loader = data.DataLoader(dataset, args.test_batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=True)
    elif args.dataset == 'td500':
        dataset = TD500Detection(args.td_root, 'val', None, None, aug=False, dim=cfg['min_dim'])
        data_loader = data.DataLoader(dataset, args.test_batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=True)
    elif args.dataset == 'coco':
        dataset = COCODetection(args.coco_root, 'test', dim=cfg['min_dim'])
        data_loader = data.DataLoader(dataset, args.test_batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=True)
    else:
        exit()

    logging.info('dataset initialize done.')

    ## setup mode

    net = build_dssd('test', cfg, ssd_dim, 2)


    logging.info('loading {}...'.format(args.resume))
    net.load_weights(args.resume)
    rpsroi_pool = RPSRoIPool(2,2,1,2,1)
    if cuda:
        net = net.cuda()
        rpsroi_pool = rpsroi_pool.cuda()
    net.eval()
    rpsroi_pool.eval()
    if os.path.exists(args.out)==False:
        os.makedirs(args.out)
    save_dir = args.out + '/' + args.resume.strip().split('_')[-1].split('.')[0] + '/'
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    seg_dir = save_dir + 'seg/'
    box_dir = save_dir + 'box/'
    res_dir = save_dir + 'res/'

    if os.path.exists(seg_dir) == False:
        os.mkdir(seg_dir)
        os.mkdir(box_dir)
        os.mkdir(res_dir)
    logging.info('eval begin')
    for i, sample in enumerate(data_loader, 0):
        img, image_name,ori_h, ori_w = sample
        # print(image_name)
        if i % 100 == 0:
            print(i, len(data_loader))
        h, w = img.size(2), img.size(3)
        if cuda:
            img = img.cuda()
        img = Variable(img)
        out, seg_pred, seg_map =net(img)
        save_name = image_name[0].split('/')[-1].split('.')[0]
        candidate_box = eval_img(out, seg_pred, seg_map, rpsroi_pool,  img, save_name, seg_dir, box_dir, vis=True)

        # format output
        if args.dataset == 'coco':
            save_name = save_name.strip().split('_')[-1]
            save_name = str(int(save_name))
        res_name = res_dir + '/' + 'res_' + save_name + '.txt'
        fp = open(res_name, 'w')
        for box in candidate_box:
            temp_x = []
            temp_y = []
            temp = []
            for j in range(len(box) - 1):
                if j % 2 == 0:
                    temp_x.append(int(box[j]*ori_w[0]/w))
                    temp.append(str(int(box[j]*ori_w[0]/w)))
                else:
                    temp_y.append(int(box[j]*ori_h[0]/h))
                    temp.append(str(int(box[j]*ori_h[0]/h)))
            if args.dataset == 'ic13':
                fp.write(','.join([str(min(temp_x)), str(min(temp_y)), str(max(temp_x)), str(max(temp_y))]) + '\n')
            elif args.dataset == 'coco':
                fp.write(','.join([str(min(temp_x)), str(min(temp_y)), str(max(temp_x)), str(max(temp_y)), str(box[-1])]) + '\n')
            else:
                fp.write(','.join(temp) + '\n')
        fp.close()


    logging.info('evaluate done')

if __name__ == '__main__':
    main()

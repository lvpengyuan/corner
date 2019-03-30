import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from ..box_utils import decode, nms
from data import train_cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = train_cfg['variance']

    def forward(self, loc_data, conf_data, prior_data, seg_data, seg_map):
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        prior_data = prior_data.cuda()
        conf_preds = conf_data.view(1, num_priors, 4, 2)
        temp = []
        for i in range(4):
            decoded_boxes = decode(loc_data[0, :, i, :], prior_data, self.variance)
            conf_scores = conf_preds[0, :, i, :].clone()
            c_mask = conf_scores[:, 1].gt(self.conf_thresh)
            scores = conf_scores[:, 1][c_mask]
            if scores.dim() == 0:
                temp.append(torch.rand(1, 6).fill_(0).cuda())
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
            temp.append(torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]], torch.rand(count, 1).fill_(i).cuda()), 1))
        res = torch.cat(temp, 0)
        return res, seg_data, seg_map

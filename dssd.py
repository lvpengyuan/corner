import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os


class DM(nn.Module):
    def __init__(self, nin, nout, ks, strid, padding):
        super(DM, self).__init__()
        self.path1 = nn.Sequential(
            nn.ConvTranspose2d(nout, nout, ks, strid, padding),
            nn.Conv2d(nout, nout, 3, 1, 1),
            nn.BatchNorm2d(nout))
        self.path2 = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.ReLU(True),
            nn.Conv2d(nout, nout, 3, 1, 1),
            nn.BatchNorm2d(nout))

    def forward(self, x1, x2):
        path1 = self.path1(x1)
        path2 = self.path2(x2)
        return F.relu(torch.mul(path1, path2))

class PM(nn.Module):
    def __init__(self, nin):
        super(PM, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(nin, 256, 1, 1))
        self.bone = nn.Sequential(
            nn.Conv2d(nin, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1))

    def forward(self, x):
        x1 = self.skip(x)
        x2 = self.bone(x)
        return F.relu(x1 + x2)

class SM(nn.Module):
    def __init__(self, nin, nout, nscale):
        super(SM, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(nin, nout, 1, 1),
            nn.BatchNorm2d(nout))
        self.bone = nn.Sequential(
            nn.Conv2d(nin, nout, 1, 1),
            nn.BatchNorm2d(nout),
            nn.ReLU(True),
            nn.Conv2d(nout, nout, 1, 1),
            nn.BatchNorm2d(nout),
            nn.ReLU(True),
            nn.Conv2d(nout, nout, 1, 1),
            nn.BatchNorm2d(nout))

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=nscale)

    def forward(self, x):
        x1 = self.skip(x)
        x2 = self.bone(x)
        return self.upsample(F.relu(x1 + x2))


class SegPred(nn.Module):
    def __init__(self, nin):
        super(SegPred, self).__init__()
        self.tail = nn.Sequential(
            nn.Conv2d(nin, nin, 1, 1),
            nn.BatchNorm2d(nin),
            nn.ReLU(True),
            nn.ConvTranspose2d(nin, nin, 2, 2),  ## 256
            nn.Conv2d(nin, nin, 3, 1, 1),
            nn.BatchNorm2d(nin),
            nn.ReLU(True),
            nn.ConvTranspose2d(nin, 4, 2, 2))
    def forward(self, xs):
        x1, x2, x3, x4, x5 = xs[0], xs[1], xs[2], xs[3], xs[4]
        fuse_feat = F.relu(x1 + x2 + x3 + x4 + x5)
        return self.tail(fuse_feat)

class DSSD(nn.Module):
    def __init__(self, phase, base, extras, head, dms, pms, sms, cfg, num_classes=2):
        super(DSSD, self).__init__()
        # DSSD network
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 512

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.dms = nn.ModuleList(dms)
        self.pms = nn.ModuleList(pms)
        self.sms = nn.ModuleList(sms)
        self.seg_pred = SegPred(32)

        if self.phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.5, 0.45)

    def forward(self, x):
        sources = list()
        feats = list()
        pm_feats = list()
        loc = list()
        conf = list()
        seg_feats = list()
        

        # apply vgg up to conv3_3 relu 256
        for k in range(16):
            x = self.vgg[k](x)
        sources.append(x)


        for k in range(16, 23):
            x = self.vgg[k](x)
        # s = self.L2Norm(x)
        # sources.append(s)
        sources.append(x)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        ## sources [conv3, conv4, fc7, conv8, conv9, conv10, conv11]

        feats.append(sources[-1])
        for i in range(6):
            feats.append(self.dms[i].forward(feats[-1], sources[-i - 2]))


        
        # seg_pred = self.seg_pred(feats[-1])
        for i in range(5):
            seg_feats.append(self.sms[i].forward(feats[i + 2]))

        seg_pred = self.seg_pred(seg_feats)


        for i in range(7):
            
            pm_feats.append(self.pms[i].forward(feats[-i - 1]))

        # apply multibox head to source layers
        for (x, l, c) in zip(pm_feats, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            seg_view = seg_pred.permute(0, 2, 3, 1).contiguous().view(-1, 1)
            output = self.detect(loc.view(loc.size(0), -1, 4, 4), self.softmax(conf.view(-1, self.num_classes)), self.priors, F.sigmoid(seg_view), F.sigmoid(seg_pred))
        else:
            seg_pred = seg_pred.permute(0, 2, 3, 1).contiguous().view(-1, 1)
            output = (
                loc.view(loc.size(0), -1, 4, 4),
                conf.view(conf.size(0), -1, 4, self.num_classes),
                self.priors,
                F.sigmoid(seg_pred),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def deconv_modules(cfg):
    dms = list()
    for i in cfg:
        dms.append(DM(i[0], i[1], i[2], i[3], i[4]))
    return dms

def predict_modules(cfg):
    pms = list()
    for i in cfg:
        pms.append(PM(i))
    return pms

def seg_modules(cfg):
    sms = list()
    for i in cfg:
        sms.append(SM(i[0], i[1], i[2]))
    return sms

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k in range(7):
        loc_layers += [nn.Conv2d(256, cfg[k] * 4 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, cfg[k] * num_classes*4, kernel_size=3, padding=1)]
        
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '512': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
}
mbox = {
    '512': [6, 4, 4, 4, 4, 4, 4],
}


dm = [[256, 256, 3, 1, 0], [256, 256, 3, 1, 0], [512, 256, 2, 2, 0], [1024, 256, 2, 2, 0], [512, 256, 2, 2, 0], [256, 256, 2, 2, 0]]

pm = [256, 256, 256, 256, 256, 256, 256]

sm = [[256, 32, 16], [256, 32, 8], [256, 32, 4], [256, 32, 2], [256, 32, 1]]

def build_dssd(phase, cfg, size=512, num_classes=2):
    base_net, extras_submodels, head = multibox(vgg(base[str(size)], 3), add_extras(extras[str(size)], 1024), mbox['512'], 2)
    return DSSD(phase, base_net, extras_submodels, head, deconv_modules(dm), predict_modules(pm), seg_modules(sm), cfg)

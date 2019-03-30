import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import train_cfg, AnnotationTransform, ICDARDetection, detection_collate, SynthDetection, TD500Detection, MLTDetection
from utils.augmentations_poly import SSDAugmentation
from utils.augmentations_synth import SynthSSDAugmentation
from utils.logger import setup_logger
from layers.modules import MultiBoxLoss
from dssd import build_dssd
import numpy as np
import time
import logging


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--name', default='synth_fuse_seg_mlt', type=str)
parser.add_argument('--version', default='v4', type=str, help='v4=512, v3=300')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', type=str, help='pretrained base model')
parser.add_argument('--ssd_dim', default=512, type=int, help='input image dim')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default='./weights/synth_fuse_seg/synth_0_30000.pth', type=str, help='Resume from checkpoint')
# parser.add_argument('--resume', default='./weights/synth_fuse_seg/icdar_final.pth', type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=60000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--optim', default='adam', type=str, help='sgd, adam, adadelta')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='weights/', type=str, help='Location to save checkpoint models')
#parser.add_argument('--root', default='../../DataSets/text_detect/',type=str,  help='Location of data root directory')
parser.add_argument('--td_root', default='/home/lpy/Datasets/TD&&TR/',type=str,  help='Location of data root directory')
parser.add_argument('--icdar_root', default='/home/lpy/Datasets/text_detect/',type=str,  help='Location of data root directory')
parser.add_argument('--synth_root', default='/data/lvpyuan/Datasets/SynthText/',type=str,  help='Location of data root directory')
parser.add_argument('--mlt_root', default='/home/lpy/Datasets/MLT/',type=str,  help='Location of data root directory')
#parser.add_argument('--root', default='/home/lvpengyuan/research/text/',type=str,  help='Location of data root directory')
parser.add_argument('--clip_grad', default=False, type=bool, help='clip grad or not')
# parser.add_argument('--mode', default='iou', type=str, help='bce, iou, focal, ohnm')
args = parser.parse_args()

args.save_folder = args.save_folder + args.name + '/'

if os.path.exists(args.save_folder) == False:
    os.mkdir(args.save_folder)
log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
setup_logger(log_file_path)
logging.info('name:%s'%(args.name))
logging.info('base learning rate: %f'%(args.lr))
logging.info('resume:%s'%(args.resume))
logging.info('optim method:%s'%(args.optim))
logging.info('data root:%s'%(args.icdar_root))
logging.info('ssd_dim:%d'%(args.ssd_dim))
logging.info('number of works:%d'%(args.num_workers))
logging.info('training batch size:%s'%(args.batch_size))
logging.info('output dir: %s'%(args.save_folder))
logging.info('clip_grad:%s'%(args.clip_grad))
# logging.info('mode:%s'%(args.mode))
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')



ssd_dim = args.ssd_dim
assert(ssd_dim == train_cfg['min_dim'][0])
means = (104, 117, 123)
batch_size = args.batch_size
max_iter = args.iterations

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

dssd_net = build_dssd('train', train_cfg, ssd_dim,2)
net = dssd_net


if args.resume:
    logging.info('Resuming training, loading {}...'.format(args.resume))
    dssd_net.load_weights(args.resume)
else:
    vgg_weights = torch.load('./weights/' + args.basenet)
    logging.info('Loading base network...')
    dssd_net.vgg.load_state_dict(vgg_weights)
    dssd_net.extras.apply(weights_init)
    dssd_net.loc.apply(weights_init)
    dssd_net.conf.apply(weights_init)
    dssd_net.pms.apply(weights_init)
    dssd_net.dms.apply(weights_init)
    dssd_net.sms.apply(weights_init)
    dssd_net.seg_pred.apply(weights_init)


net = torch.nn.DataParallel(dssd_net)
if args.cuda:
    net = net.cuda()


if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    logging.info('model will be optimed by sgd')
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    logging.info('model will be optimed by adam')
elif args.optim == 'adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    logging.info('model will be optimed by adam')

num_classes = 2
criterion = MultiBoxLoss(num_classes, 0.5, 3, args.cuda)


def train():
    net.train()
    logging.info('Loading Dataset...')
    # dataset = SynthDetection(args.synth_root, SynthSSDAugmentation(
    #     ssd_dim, means), AnnotationTransform())
    # data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
    #                               shuffle=False, collate_fn=detection_collate, pin_memory=True)
    # logging.info('Training SSD_CBTD_BOX on Synth')
    # for i in range(1):
    #     for j, batch_samples in enumerate(data_loader):
    #         images, targets, segs  = batch_samples
    #         gts = []
    #         if args.cuda:
    #             images = Variable(images.cuda())
    #             segs = Variable(segs.cuda())
    #             for sample in targets:
    #                 temp = []
    #                 for item in sample:
    #                     temp.append(Variable(item.cuda(), volatile=True))
    #                 gts.append(temp)
    #             targets = gts
    #         else:
    #             images = Variable(images)
    #             segs = Variable(segs)
    #             for sample in targets:
    #                 temp = []
    #                 for item in samples:
    #                     temp.append(Variable(item, volatile=True))
    #                 gts.append(temp)
    #             targets = gts
    #         # zero_grad
    #         optimizer.zero_grad()

    #         # forward
    #         t0 = time.time()
    #         out = net(images)
    #         loss_l, loss_c, loss_s = criterion(out, targets, segs)
    #         loss_s = loss_s*10
    #         loss = loss_l + loss_c + loss_s
    #         loss.backward()
    #         if args.clip_grad == True:
    #             torch.nn.utils.clip_grad_norm(net.parameters(), 0.25)
    #         optimizer.step()
    #         t1 = time.time()
    #         if j % 10 == 0:
    #             logging.info('synth-> iter:%6d epoch:%3d loss:%.6f loss_l:%.6f loss_c:%.6f loss_s:%.6f time:%.4f'%(j, i, loss.data[0], loss_l.data[0], loss_c.data[0], loss_s.data[0], (t1 - t0)))
    #         if j % 5000 == 0:
    #             logging.info('Saving state, epoch: %d iter:%d'%(i, j))
    #             torch.save(dssd_net.state_dict(), 'weights/' + args.name + '/synth_' +
    #                        repr(i) + '_' + repr(j) + '.pth')
    # torch.save(dssd_net.state_dict(), 'weights/' + args.name + '/synth' + '.pth')

    # logging.info('Training SSD_CBTD_BOX on ICDAR')
    # dataset = ICDARDetection(args.icdar_root, 'train', SSDAugmentation(
    #     train_cfg['min_dim'], means), AnnotationTransform(), '13&&15', dim=train_cfg['min_dim'])
    # data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
    #                                shuffle=True, collate_fn=detection_collate, pin_memory=True)
    
    #logging.info('Training SSD_CBTD_BOX on TD500')
    #dataset = TD500Detection(args.td_root, 'train', SSDAugmentation(
    #(512, 512), means), AnnotationTransform(), aug=True)
    #data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
    #                              shuffle=True, collate_fn=detection_collate, pin_memory=True)
    logging.info('Training SSD_CBTD_BOX on MLT')
    dataset = MLTDetection(args.mlt_root, 'train', SSDAugmentation(
        (512, 512), means), AnnotationTransform())
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                 shuffle=True, collate_fn=detection_collate, pin_memory=True)

    for i in range(200):
        for j, batch_samples in enumerate(data_loader):
            images, targets, segs  = batch_samples
            gts = []
            if args.cuda:
                images = Variable(images.cuda())
                segs = Variable(segs.cuda())
                for sample in targets:
                    temp = []
                    for item in sample:
                        temp.append(Variable(item.cuda(), volatile=True))
                    gts.append(temp)
                targets = gts
            else:
                images = Variable(images)
                segs = Variable(segs)
                for sample in targets:
                    temp = []
                    for item in samples:
                        temp.append(Variable(item, volatile=True))
                    gts.append(temp)
                targets = gts
            # zero_grad
            optimizer.zero_grad()

            # forward
            t0 = time.time()
            out = net(images)
            loss_l, loss_c, loss_s = criterion(out, targets, segs)
            loss_s = loss_s*10
            loss = loss_l + loss_c + loss_s
            loss.backward()
            if args.clip_grad == True:
                torch.nn.utils.clip_grad_norm(net.parameters(), 0.25)
            optimizer.step()
            t1 = time.time()
            if j % 10 == 0:
                logging.info('icdar-> iter:%6d epoch:%3d loss:%.6f loss_l:%.6f loss_c:%.6f loss_s:%.6f time:%.4f'%(j, i, loss.data[0], loss_l.data[0], loss_c.data[0], loss_s.data[0], (t1 - t0)))
        if i % 5 == 0:
            logging.info('Saving state, epoch: %d iter:%d'%(i, j))
            torch.save(dssd_net.state_dict(), 'weights/' + args.name + '/mlt_' +
                       repr(i)  + '.pth')
    torch.save(dssd_net.state_dict(), 'weights/' + args.name + '/mlt_' + '.pth')


if __name__ == '__main__':
    train()

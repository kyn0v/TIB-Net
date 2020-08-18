#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


# Adapt from ->
# --------------------------------------------------------
# '''
# EXTD Copyright (c) 2019-present NAVER Corp. MIT License
# '''
# --------------------------------------------------------
# <- Written by kyn0v

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from config import cfg
from backbone.tibnet import build_tibnet
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='TIB-Net Training With Pytorch')
parser.add_argument('--dataset',
                    default='voc',
                    choices=['voc'],
                    help='Train target')
parser.add_argument('--batch_size',
                    default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=2e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder',
                    default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrained', default='./weights/mobileFacenet.pth',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

args = parser.parse_args()


def compute_flops(model, image_size):
    import torch.nn as nn
    flops = 0.
    input_size = image_size
    for m in model.modules():
        if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
            input_size = input_size / 2.
        if isinstance(m, nn.Conv2d):
            if m.groups == 1:
                flop = (input_size[0] / m.stride[0] * input_size[1] / m.stride[1]
                        ) * m.kernel_size[0] ** 2 * m.in_channels * m.out_channels
            else:
                flop = (input_size[0] / m.stride[0] * input_size[1] / m.stride[1]) * m.kernel_size[0] ** 2 * (
                    (m.in_channels/m.groups) * (m.out_channels/m.groups) * m.groups)
            flops += flop
            if m.stride[0] == 2:
                input_size = input_size / 2.

    return flops / 1000000000., flops / 1000000


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

save_folder = args.save_folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print('load dataset...')
train_dataset, val_dataset = dataset_factory(args.dataset)
train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)
val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)

min_loss = np.inf

start_epoch = 0
tibnet = build_tibnet('train', cfg.NUM_CLASSES)
net = tibnet
print(net)

gflops, mflops = compute_flops(net, np.array([cfg.INPUT_SIZE, cfg.INPUT_SIZE]))
print('params: %d, flops: %.2f GFLOPS, %.2f MFLOPS, ImageSize: %d' %
      (sum([p.data.nelement() for p in net.parameters()]), gflops, mflops, cfg.INPUT_SIZE))


if args.resume:
    print('load checkpoint {} ...'.format(args.resume))
    start_epoch = net.load_weights(args.resume)

else:
    try:
        _weights = torch.load(args.pretrained)
        print('load pretrained weight....')
        net.base.load_state_dict(_weights['state_dict'], strict=False)
    except:
        print('initialize base network....')
        net.base.apply(net.weights_init)

if args.cuda:
    net = net.cuda()
    cudnn.benckmark = True

if not args.resume:
    print('initialize network...')
    tibnet.loc.apply(tibnet.weights_init)
    tibnet.conf.apply(tibnet.weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)


print('args:\n', args)


def train():
    step_index = 0
    iteration = 0
    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.cuda:
                images = Variable(images.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda())
                               for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c  # stress more on loss_l
            loss_add = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            losses += loss_add.item()

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                print("[epoch:{}][iter:{}][lr:{:.5f}] loss_class {:.8f} - loss_reg {:.8f} - total {:.8f}".format(
                    epoch, iteration, args.lr, loss_c.item(), loss_l.item(), tloss
                ))
            iteration += 1
        # You can adjust the evaluation interval here:
        # if epoch%{INTERVAL}==0:
        val(epoch)
        if iteration == cfg.MAX_STEPS:
            break


def val(epoch):
    net.eval()
    loc_loss = 0
    conf_loss = 0
    step = 0

    with torch.no_grad():
        t1 = time.time()
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.cuda:
                images = Variable(images.cuda())
                with torch.no_grad():
                    targets = [ann.cuda() for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            out = net(images)
            loss_l, loss_c = criterion(out, targets)

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            step += 1

        tloss = (loc_loss + 3 * conf_loss) / step
        t2 = time.time()
        print('Timer: %.4f' % (t2 - t1))
        print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

        states = {
            'epoch': epoch,
            'weight': tibnet.state_dict(),
        }

        global min_loss
        if tloss < min_loss:
            file_with_epoch = 'best_weight.pth'
            torch.save(states, os.path.join(save_folder, file_with_epoch))
            min_loss = tloss

        if(epoch % 10 == 0):
            print('save checkpoint of epoch', epoch)
            file = 'epoch_{}.pth'.format(epoch)
            torch.save(states, os.path.join(save_folder, file))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()

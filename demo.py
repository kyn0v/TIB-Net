#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Detect UAV from pictures to be tested, and save the result pictures.

# Adapt from ->
# --------------------------------------------------------
# '''
# EXTD Copyright (c) 2019-present NAVER Corp. MIT License
# '''
# --------------------------------------------------------
# <- Written by kyn0v


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import cv2
import time
import numpy as np
from PIL import Image

from config import cfg
from backbone.tibnet import build_tibnet
from torch.autograd import Variable
from data.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='TIB-Net demo')
parser.add_argument('--image_dir', type=str, default='./result/demo/img/',
                    help='Directory for detect result')
parser.add_argument('--save_dir', type=str, default='./result/demo/img_det/',
                    help='Directory for detect result')
parser.add_argument('--weight', type=str, help='weight file', required=True)
parser.add_argument('--thresh', default=0.5, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


if not os.path.exists(args.image_dir):
    os.makedirs(args.image_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect(net, img_path, thresh, wid, hei):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')
    img = np.array(img)
    height, width, _ = img.shape
    if wid > 0 and hei > 0:
        image = cv2.resize(img, (wid, hei))
    else:
        max_im_shrink = np.sqrt(1700 * 1200 / (img.shape[0] * img.shape[1]))
        image = cv2.resize(img, None, None, fx=max_im_shrink,
                           fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]
    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    with torch.no_grad():
        t1 = time.time()
        detections = net(x).data
        t2 = time.time()
        print('detect:{} timer:{}'.format(os.path.basename(img_path), t2 - t1))
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
                j += 1
                cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
                conf = "{:.3f}".format(score)
                point = (int(left_up[0]), int(left_up[1] - 5))
                cv2.putText(img, conf, point,
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)
    return t2-t1

if __name__ == '__main__':
    net = build_tibnet('test', cfg.NUM_CLASSES)
    weight_file = os.path.join(args.weight)
    print("load weight file from {}.".format(weight_file))
    net.load_weights(weight_file)
    net.eval()
    if use_cuda:
        net.cuda()
        cudnn.benckmark = True
    img_path = args.image_dir
    img_list = [os.path.join(img_path, x) for x in os.listdir(img_path) if x.endswith('jpg')]
    for path in img_list:
        dt = detect(net, path, args.thresh, -1, -1)

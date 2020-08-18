#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Generate test result in '.txt' & '.pkl' format.

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import xml.etree.ElementTree as ET
import _pickle as cPickle

import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

import cv2
import time
import pickle
from PIL import Image

from data.vocdataset import VOCDetection, VOCAnnotationTransform
from data.augmentations import to_chw_bgr
from backbone.tibnet import build_tibnet
from config import cfg


labelmap = ['uav']
parser = argparse.ArgumentParser(
    description='TIB-Net test result(.txt) generator')
parser.add_argument('--weight', required = True)
args = parser.parse_args()


class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def write_voc_results_file(all_boxes, dataset, filename):
    label_map = ['uav']
    for cls_ind, cls in enumerate(labelmap):
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                print(im_ind, index)
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def test_net(save_folder, net, dataset, thresh=0.5):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    _t = {'im_detect': Timer(), 'misc': Timer()}

    output_dir = os.path.join(save_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timesum = 0
    for i in range(num_images):
        img = dataset.pull_image(i)
        h, w, _ = img.shape

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

        _t['im_detect'].tic()
        with torch.no_grad():
            detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(thresh).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets
            fin_mask = np.where(scores > 0.6)[0]
            bboxes = boxes.cpu().numpy()[fin_mask]
            scores = scores[fin_mask]
            for k in range(len(scores)):
                leftup = (int(bboxes[k][0]), int(bboxes[k][1]))
                right_bottom = (int(bboxes[k][2]), int(bboxes[k][3]))
                cv2.rectangle(img, leftup, right_bottom, (0, 255, 0), 2)

        print('uav_detect: {:d}/{:d} {:.3f}s'.format(i +
                                                    1, num_images, detect_time))
        timesum += detect_time

    det_file_pkl = os.path.join(save_folder, 'result.pkl')
    with open(det_file_pkl, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    det_file_txt = os.path.join(save_folder, 'result.txt')
    write_voc_results_file(all_boxes, dataset, det_file_txt)
    print("detection result has been saved in '.pkl' & '.txt' format.")
    print("the average time of detection is {:.3f}s".format(
        timesum/num_images))


if __name__ == '__main__':
    net = build_tibnet('test', cfg.NUM_CLASSES)
    weight_file = os.path.join(args.weight)
    print("load weight file from {}.".format(weight_file))
    net.load_state_dict(torch.load(weight_file)['weight'])
    net.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        cudnn.benckmark = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('network has been loaded.')
    dataset = VOCDetection(cfg.VOC.HOME,
                           target_transform=VOCAnnotationTransform(),
                           mode='test')
    print("dataset has been loads.")
    save_folder = 'result/detection'
    print("the result would be saved to {}.".format(save_folder))
    test_net(save_folder, net, dataset, 0.5)

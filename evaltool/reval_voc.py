#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Adapt from ->
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# <- Written by kyn0v

"""Reval = re-eval. Re-evaluate saved detections."""

import os
import sys
import argparse
import numpy as np
import pickle as cPickle
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import precision_recall_curve
from itertools import cycle

from voc_eval import voc_eval


def do_python_eval(data_path, det_file_path, classes, det_dir, output_dir, ovthresh):
    anno_path = os.path.join(data_path, 'Annotations', '{:s}.xml')
    test_file = os.path.join(data_path, 'test.txt')
    cache_dir = os.path.join('../result/evaluation')
    aps = []
    # set the evaluation rules
    year = 2007
    use_07_metric = True if int(year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        rec, prec, ap = voc_eval(
            det_file_path, anno_path, test_file, cls, cache_dir, ovthresh,
            use_07_metric=use_07_metric)
        aps += [ap]
        pl.plot(rec, prec, lw=2,
                label='Precision-recall curve of class {} (area = {:.4f})'''.format(cls, ap))
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    # ------------rough drawing of PR-Curve------------
    # pl.xlabel('Recall')
    # pl.ylabel('Precision')
    # plt.grid(True)
    # pl.ylim([0.0, 1.05])
    # pl.xlim([0.0, 1.0])
    # pl.title('Precision-Recall')
    # pl.legend(loc="upper right")
    # plt.show()

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('MAP:{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')


if __name__ == '__main__':
    det_result_dir = '../result/detection'
    det_file_path = os.path.join(det_result_dir, 'result.txt')
    output_dir = '../result/evaluation'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset_path = '../dataset'
    classes = ['uav']
    print("Start evaluation...")
    do_python_eval(dataset_path, det_file_path,
                   classes, det_result_dir, output_dir, ovthresh=0.5)

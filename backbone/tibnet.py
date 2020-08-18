#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Adapt from ->
# -------------------------------------------------------
# '''
# EXTD Copyright (c) 2019-present NAVER Corp. MIT License
# '''
# -------------------------------------------------------
# <- Written by kyn0v.


import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from layers.modules.l2norm import L2Norm
from layers.functions.prior_box import PriorBox
from layers.functions.detection import Detect
from torch.autograd import Variable

from layers import *
from config import cfg
import numpy as np
from . import mobilefacenet


def upsample(in_channels, out_channels):  # should use F.inpterpolate
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                  stride=1, padding=1, groups=in_channels, bias=False),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


    def forward(self, x):
        raw = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return raw*self.sigmoid(x)

class TibNet(nn.Module):
    """
    Desc:
        Single Shot Multibox Architecture
        The network is composed of a base VGG network followed by the
        added multibox conv layers.  Each multibox layer branches into
            1) conv2d for class conf scores
            2) conv2d for localization predictions
            3) associated priorbox layer to produce default bounding
            boxes specific to the layer's feature map size.
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, phase, basenet, head, num_classes):
        super(TibNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.sa = mobilefacenet.SpatialAttention()
        self.base = nn.ModuleList(basenet)
        self.upfeat = []
        for it in range(5):
            self.upfeat.append(upsample(in_channels=64, out_channels=64))
        self.upfeat = nn.ModuleList(self.upfeat)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def forward(self, x):
        """
        Desc:
            Applies network layers and ops on input image(s) x.
        """
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()

        # cyclic pathway
        for k in range(6):
            if(k == 2):
                before = x
            x = self.base[k](x)
        s1 = x
        for k in range(2, 6):
            if(k == 3):
                x += nn.Conv2d(64, 64, kernel_size=(3, 3),
                               stride=2, padding=1).cuda()(before)
            x = self.base[k](x)
        s2 = x
        for k in range(2, 6):
            if(k == 3):
                x += nn.Conv2d(64, 64, kernel_size=(3, 3),
                               stride=4, padding=1).cuda()(before)
            x = self.base[k](x)
        s3 = x
        for k in range(2, 6):
            if(k == 3):
                x += nn.Conv2d(64, 64, kernel_size=(3, 3),
                               stride=8, padding=1).cuda()(before)
            x = self.base[k](x)
        s4 = x
        for k in range(2, 6):
            if(k == 3):
                x += nn.Conv2d(64, 64, kernel_size=(3, 3),
                               stride=16, padding=1).cuda()(before)
            x = self.base[k](x)
        s5 = x
        for k in range(2, 6):
            if(k == 3):
                x += nn.Conv2d(64, 64, kernel_size=(3, 3),
                               stride=32, padding=1).cuda()(before)
            x = self.base[k](x)
        s6 = x

        # FPN-like
        sources.append(s6)
        u1 = self.upfeat[0](F.interpolate(
            s6, size=(s5.size()[2], s5.size()[3]), mode='bilinear')) + s5  # 10x10
        sources.append(u1)
        u2 = self.upfeat[1](F.interpolate(
            u1, size=(s4.size()[2], s4.size()[3]), mode='bilinear')) + s4  # 20x20
        sources.append(u2)
        u3 = self.upfeat[2](F.interpolate(
            u2, size=(s3.size()[2], s3.size()[3]), mode='bilinear')) + s3  # 40x40
        sources.append(u3)
        u4 = self.upfeat[3](F.interpolate(
            u3, size=(s2.size()[2], s2.size()[3]), mode='bilinear')) + s2  # 80x80
        sources.append(u4)
        u5 = self.upfeat[4](F.interpolate(
            u4, size=(s1.size()[2], s1.size()[3]), mode='bilinear')) + s1  # 160x160
        sources.append(u5)
        sources = sources[::-1]  # reverse order

        # SSD-like: apply multibox head to source layers
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])
        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)
        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())
        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())
        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]
        self.priorbox = PriorBox(size, features_maps, cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight.data)
            if torch.is_tensor(m.bias):
                m.bias.data.zero_()

def basenet():
    net = mobilefacenet.MobileFaceNet64_SA()
    return nn.ModuleList(net.features)

def multibox(basenet, num_classes):
    loc_layers = []
    conf_layers = []
    net_source = [1, 2, 3, 4, 5]
    # base[0]--base[5]
    feature_dim = []
    feature_dim += [basenet[0][-3].out_channels]
    for idx in net_source:
        feature_dim += [basenet[idx].conv[-2].out_channels]
    
    loc_layers += [nn.Conv2d(feature_dim[0], 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(feature_dim[0], 3 + (num_classes - 1), kernel_size=3, padding=1)]
    for v in feature_dim[1:]:
        loc_layers += [nn.Conv2d(v, 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v,num_classes, kernel_size=3, padding=1)]
    return basenet[:6], (loc_layers, conf_layers)

def build_tibnet(phase, num_classes=2):
    base_, head_ = multibox(basenet(), num_classes)
    return TibNet(phase, base_, head_, num_classes)
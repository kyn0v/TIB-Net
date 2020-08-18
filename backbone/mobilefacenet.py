#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import torch.nn as nn
import math
import torch
import torch.nn.functional as F


def conv_bn(inp, oup, stride, k_size=3):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU()

    )


class gated_conv1x1(nn.Module):
    def __init__(self, inc=128, outc=128):
        super(gated_conv1x1, self).__init__()
        self.inp = int(inc/2)
        self.oup = int(outc/2)
        self.conv1x1_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_1 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)
        self.conv1x1_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=False)
        self.gate_2 = nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True)

    def forward(self, x):
        x_1 = x[:, :self.inp, :, :]
        x_2 = x[:, self.inp:, :, :]

        a_1 = self.conv1x1_1(x_1)
        g_1 = F.sigmoid(self.gate_1(x_1))

        a_2 = self.conv1x1_2(x_2)
        g_2 = F.sigmoid(self.gate_2(x_2))

        ret = torch.cat((a_1*g_1, a_2*g_2), 1)

        return ret


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        raw = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return raw*self.sigmoid(x)


class InvertedResidual_dwc(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_dwc, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = []
        if expand_ratio == 1:
            self.conv.append(nn.Conv2d(inp, hidden_dim, kernel_size=(
                3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        else:
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(
                3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
            return out
        else:
            out = self.conv(x)
            return out


class InvertedResidual_dwc_attention(nn.Module):
    """
    Desc:
        InvertedResidualBlock with spatial attention.
    """

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_dwc_attention, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = []
        if expand_ratio == 1:
            self.conv.append(nn.Conv2d(inp, hidden_dim, kernel_size=(
                3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(SpatialAttention())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))
        else:
            self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(
                3, 3), stride=stride, padding=1, groups=hidden_dim))
            self.conv.append(nn.BatchNorm2d(hidden_dim))
            self.conv.append(nn.PReLU())
            self.conv.append(SpatialAttention())
            self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            self.conv.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
            return out
        else:
            out = self.conv(x)
            return out


class MobileFaceNet64_SA(nn.Module):
    def __init__(self, embedding_size=128, input_size=224, width_mult=1.):
        super(MobileFaceNet64_SA, self).__init__()
        block_dwc = InvertedResidual_dwc
        block_dwc_attention = InvertedResidual_dwc_attention
        input_channel = 64
        last_channel = 256
        # Note: here is the original network setting, but we only the first five blocks: base[1]--base[5]
        interverted_residual_setting = [
            # t, c, n, s
            [1, 64, 1, 1],  # base[1]
            [2, 64, 2, 1],  # base[2], base[3]
            [4, 64, 2, 2],  # base[4] s=1, base[5] s=2
            [2, 64, 2, 1],
            [4, 64, 5, 1],
            [2, 64, 2, 2],
            [2, 64, 6, 2],
        ]
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel  # 256
        self.features = [conv_bn(3, input_channel, 2)]  # base[0]

        cnt = 0  # Only add attention module in the first two IR block
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == n - 1:  # Only reduce the size of featuremap in the last
                    if cnt < 2:
                        self.features.append(block_dwc_attention(
                            input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(
                            block_dwc(input_channel, output_channel, s, expand_ratio=t))
                else:
                    if cnt < 2:
                        self.features.append(block_dwc_attention(
                            input_channel, output_channel, 1, expand_ratio=t))
                    else:
                        self.features.append(
                            block_dwc(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
            cnt += 1

        self.features.append(gated_conv1x1(input_channel, self.last_channel))
        self.features_sequential = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x):
        x = self.features_sequential(x).view(-1, 256*4)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

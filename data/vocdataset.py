import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
from PIL import ImageDraw, Image

from config import cfg
from .augmentations import preprocess


class VOCAnnotationTransform(object):
    def __call__(self, target, width, height):
        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            label_idx = 1
            bndbox.append(label_idx)
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            res += [bndbox]  # [label_ind, xmin, ymin, xmax, ymax]
            # img_id = target.find('filename').text[:-4]
        return res  # [[label_ind, xmin, ymin, xmax, ymax], ... ], muti-objects


class VOCDetection(data.Dataset):
    def __init__(self, root,
                 target_transform=VOCAnnotationTransform(),
                 mode='train'):

        self.root = root
        self.mode = mode
        self.target_transform = target_transform
        self._annopath = osp.join(root, 'Annotations', '%s.xml')
        self._imgpath = osp.join(root, 'JPEGImages', '%s.jpg')
        self.ids = []
        for line in open(osp.join(root, mode+'.txt')):
            self.ids.append(line.strip())

    def __getitem__(self, index):
        im, gt = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        while True:
            img_id = self.ids[index]
            img_path = self._imgpath % img_id
            target = ET.parse(self._annopath % img_id).getroot()
            img = Image.open(img_path)

            if img.mode == 'L':
                img = img.convert('RGB')

            width, height = img.size
            if self.target_transform is not None:  # 将target转换为指定格式
                target = self.target_transform(target, width, height)

            bbox_labels = target
            target = np.array(target)
            if target.ndim != 2:
                index = random.randrange(0, len(self.ids))
                continue

            img, sample_labels = preprocess(img, bbox_labels, self.mode)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                # from [conf, loc] to [loc, conf]
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))
                # ensure xmax>xmin, ymax>ymin
                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break
            else:
                index = random.randrange(0, len(self.ids))
        return torch.from_numpy(img), target

    def pull_image(self, index):
        """
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        """
        img_id = self.ids[index]
        img_path = self._imgpath % img_id
        img = Image.open(img_path)
        if img.mode == 'L':
            img.convert('RGB')
        img = np.array(img)
        return img

    def pull_anno(self, index):
        """
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id, gt

    def pull_tensor(self, index):
        """
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

# TIB-Net: Drone Detection Network With Tiny Iterative Backbone

# Introduction

This project hosts the code for implementing the TIB-Net for drone detection.
>[TIB-Net: Drone Detection Network With Tiny Iterative Backbone](https://ieeexplore.ieee.org/document/9141228)  
>H. Sun, J. Yang, J. Shen, D. Liang, L. Ning-Zhong and H. Zhouin  
>In: IEEE Access, vol. 8, pp. 130697-130707, 2020

![avatar](https://github.com/kyn0v/TIB-Net/blob/master/result/demo/img_det/display.jpg)

# Installation

## Requirements

This implementation is based on [EXTD](https://github.com/clovaai/EXTD_Pytorch), and the [Environment Requirement](https://github.com/clovaai/EXTD_Pytorch#requirement) is basically the same. The following lists our current experimental environment:
* Ubuntu 20.04.1 LTS
* TITAN V
* python 3.6
* pytorch 1.0.0
* CUDA 9.0
* GCC 6

## Install TIB-Net

a. Create a conda virtual environment and activate it.
```shell
conda create -n tibnet python=3.6  
conda activate tibnet
```
b. Clone the TIB-Net repository.
```shell
git clone https://github.com/kyn0v/TIB-Net.git
cd TIB-Net
```

c. Install build requirements
```shell
pip install -r requirements.txt
```

# Getting Started

## Prepare datasets

It is recommended to symlink the dataset root to $TIB-Net/data. If your folder structure is different, you may need to change the corresponding paths in `./config.py`.
```
├── backbone
├── config.py
├── data
├── dataset 
│   ├── Annotations
│   │   ├── 000001.xml
│   │   ├── ...
│   │   └── 002850.xml
│   ├── JPEGImages
│   │   ├── 000001.jpg
│   │   ├── ...
│   │   └── 002850.jpg
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── result
│   ├── demo
│   │   ├── img
│   │   └── img_det
│   ├── detection
│   └── evaluation
├── demo.py
└── ...
```
In addition, our collected drone dataset has been uploaded to Google Drive, and you can download [HERE](https://drive.google.com/drive/folders/1ro-S2lwBmn83HLSppr5i-hBHLlYLAobg?usp=sharing).
## Inference with pretrained models

You can use the following commands to test a dataset.
```shell
python test.py --weight ${WEIGHT_FILE}
```
Then the result file (in '.pkl' & 'txt' format) would be save in `./result/detection/`.

## Image demo

We provide a demo script to test all images in specified path(default:`./result/demo/img/`), and save the annotated images(default:`./result/demo/img_det/`):
```shell
python demo.py [--image_dir ${IMGDIR}] [--save_dir ${SAVEDIR}] --weight ${WEIGHT_FILE} [--thresh ${THRESH}]
```
## Train a model

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by modifying the training part in `./train.py`.
```shell
python demo.py [--batch_size ${BATCHSIZE}] [--resume ${CHECKPOINT}] [--num_workers ${WORKSNUM}] [--lr ${LEARNINGRATE}]
```

**NOTE**:
- We only support single-GPU training and testing so far.
- `--resume` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint.

## Useful tool

We provide a useful evaltool under `./evaltool/` directory. You can compute mAP and save the evaluation result would be saved in `./result/evaluation/`.
```shell
cd evaltool/
python reval_voc.py
```
Then the result file (in '.pkl' format) would be save in `./result/evaluation/`. Moreover, the function about drawing simple PR curve was commented, and you can activate the feature by modifying corresponding file. 

# Acknowledgement
We appreciate all the contributors who open source code and promote community development, and we wish that our work could also inspire other researchers.

# References
* [TIB-Net: Drone Detection Network With Tiny Iterative Backbone](https://ieeexplore.ieee.org/document/9141228)
* [EXTD_Pytorch](https://github.com/clovaai/EXTD_Pytorch)
* [S3FD.pytorch](https://github.com/yxlijun/S3FD.pytorch)

# Citations
```
@ARTICLE{9141228,
  author={H. {Sun} and J. {Yang} and J. {Shen} and D. {Liang} and L. {Ning-Zhong} and H. {Zhou}},
  journal={IEEE Access}, 
  title={TIB-Net: Drone Detection Network With Tiny Iterative Backbone}, 
  year={2020},
  volume={8},
  number={},
  pages={130697-130707},}
```

------
*May the force be with you~*
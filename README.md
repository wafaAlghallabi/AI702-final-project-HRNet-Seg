## Introduction
This repository is based on:
1. https://github.com/HRNet/HRNet-Semantic-Segmentation, the official code of [high-resolution representations for Semantic Segmentation](https://arxiv.org/abs/1904.04514). 
2. https://github.com/HRNet/Lite-HRNet, the official code of [Lite-HRNet: A Lightweight High-Resolution Network] (https://arxiv.org/abs/2104.06403).
3. https://github.com/shachoi/HANet, the official code of [Cars Can't Fly Up in the Sky: Improving Urban-Scene Segmentation via Height-Driven Attention Networks] (https://arxiv.org/abs/2003.05128)

## Quick start
1. Clone this repository and install the dependencies: pip install -r requirements.txt

### Data preparation
Download the [Cityscapes](https://www.cityscapes-dataset.com/)

Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
├── lip
│   ├── TrainVal_images
│   │   ├── train_images
│   │   └── val_images
│   └── TrainVal_parsing_annotations
│       ├── train_segmentations
│       ├── train_segmentations_reversed
│       └── val_segmentations
├── pascal_ctx
│   ├── common
│   ├── PythonAPI
│   ├── res
│   └── VOCdevkit
│       └── VOC2010
├── cocostuff
│   ├── train
│   │   ├── image
│   │   └── label
│   └── val
│       ├── image
│       └── label
├── ade20k
│   ├── train
│   │   ├── image
│   │   └── label
│   └── val
│       ├── image
│       └── label
├── list
│   ├── cityscapes
│   │   ├── test.lst
│   │   ├── trainval.lst
│   │   └── val.lst
│   ├── lip
│   │   ├── testvalList.txt
│   │   ├── trainList.txt
│   │   └── valList.txt
````

## Segmentation models
All the models files can be found in `lib/models`.
The default file of HRNet model is `lib/models/seg_hrnet.py`.

For adding the HANet, we modified the default file `lib/models/seg_hrnt_hanet.py`, and we added the following files:
1. `lib/models/deepv3.py`.
2. `lib/models/HANet.py`.
3. `lib/models/PosEmbedding_HANet.py`.
4. `lib/models/mynn_HANet.py`.

For lite-HRNet, we added the following files:
1. `lib/models/litehrnet.py`.
2. `lib/core/resnet.py`.


## Training

````
#### Training

1. Specify the training script in `tools/`.
  a. For trainig the (hrnet) models use the `tools/train.py`.
  b. For trainig the (lite-hrnet) models use the `tools/lite-train.py`.
  
2. Specify the configuration file (.yaml) of the experiment in `experiments/cityscapes`.


Here are some examples scripts to start training:

Training HRNet-W32 on Cityscapes with a batch size of 8:
````bash
python tools/train.py --cfg experiments/cityscapes/seg_hrnet_w32.yaml
````
Training Lite-HRNet-W32 on Cityscapes with a batch size of 8:
````bash
python tools/lite_train.py --cfg experiments/cityscapes/seg_lite_hrnet_w32.yaml
````

## Citation
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}

@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}

@InProceedings{Choi_2020_CVPR,
author = {Choi, Sungha and Kim, Joanne T. and Choo, Jaegul},
title = {Cars Can't Fly Up in the Sky: Improving Urban-Scene Segmentation via Height-Driven Attention Networks},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)

## Acknowledgement
We adopt sync-bn implemented by [InplaceABN](https://github.com/mapillary/inplace_abn) for PyTorch 0.4.1 experiments and the official 
sync-bn provided by PyTorch for PyTorch 1.10 experiments.

We adopt data precosessing on the PASCAL-Context dataset, implemented by [PASCAL API](https://github.com/zhanghang1989/detail-api).

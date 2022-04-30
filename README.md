## Introduction
This repository is based on the , https://github.com/HRNet/HRNet-Semantic-Segmentation, the official code of [high-resolution representations for Semantic Segmentation](https://arxiv.org/abs/1904.04514). 

## Segmentation models
The models are initialized by the weights pretrained on the ImageNet. ''Paddle'' means the results are based on PaddleCls pretrained HRNet models.
You can download the pretrained models from  https://github.com/HRNet/HRNet-Image-Classification. *Slightly different, we use align_corners = True for upsampling in HRNet*.

1. Performance on the Cityscapes dataset. The models are trained and tested with the input size of 512x1024 and 1024x2048 respectively.
If multi-scale testing is used, we adopt scales: 0.5,0.75,1.0,1.25,1.5,1.75.

| model | Train Set | Test Set | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | Train | Val | No | No | No | 80.9 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_cs_8090_torch11.pth)/[BaiduYun(Access Code:pmix)](https://pan.baidu.com/s/1KyiOUOR0SYxKtJfIlD5o-w)|
| HRNetV2-W48 + OCR | Train | Val | No | No | No | 81.6 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cs_8162_torch11.pth)/[BaiduYun(Access Code:fa6i)](https://pan.baidu.com/s/1BGNt4Xmx3yfXUS8yjde0hQ)|
| HRNetV2-W48 + OCR | Train + Val | Test | No | Yes | Yes | 82.3 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cs_trainval_8227_torch11.pth)/[BaiduYun(Access Code:ycrk)](https://pan.baidu.com/s/16mD81UnGzjUBD-haDQfzIQ)|
| HRNetV2-W48 (Paddle) | Train | Val | No | No | No | 81.6 | ---|
| HRNetV2-W48 + OCR (Paddle) | Train | Val | No | No | No | --- | ---|
| HRNetV2-W48 + OCR (Paddle) | Train + Val | Test | No | Yes | Yes | --- | ---|

## Quick start
### Install
1. For LIP dataset, install PyTorch=0.4.1 following the [official instructions](https://pytorch.org/). For Cityscapes and PASCAL-Context, we use PyTorch=1.1.0.
2. `git clone https://github.com/HRNet/HRNet-Semantic-Segmentation $SEG_ROOT`
3. Install dependencies: pip install -r requirements.txt

If you want to train and evaluate our models on PASCAL-Context, you need to install [details](https://github.com/zhanghang1989/detail-api).
````bash
pip install git+https://github.com/zhanghang1989/detail-api.git#subdirectory=PythonAPI
````

### Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/)

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

### Train and Test

````
#### Training

Just specify the configuration file for `tools/train.py`.

For example, train the HRNet-W48 on Cityscapes with a batch size of 12 on 4 GPUs:
````bash
$PY_CMD tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````
For example, train the HRNet-W48 + OCR on Cityscapes with a batch size of 12 on 4 GPUs:
````bash
$PY_CMD tools/train.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````

Note that we only reproduce HRNet+OCR on LIP dataset using PyTorch 0.4.1. So we recommend to use PyTorch 0.4.1 if you want to train on LIP dataset.

#### Testing

For example, evaluating HRNet+OCR on the Cityscapes validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     TEST.MODEL_FILE hrnet_ocr_cs_8162_torch11.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating HRNet+OCR on the Cityscapes test set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET list/cityscapes/test.lst \
                     TEST.MODEL_FILE hrnet_ocr_trainval_cs_8227_torch11.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating HRNet+OCR on the PASCAL-Context validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/pascal_ctx/seg_hrnet_ocr_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200.yaml \
                     DATASET.TEST_SET testval \
                     TEST.MODEL_FILE hrnet_ocr_pascal_ctx_5618_torch11.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.FLIP_TEST True
````
Evaluating HRNet+OCR on the LIP validation set with flip testing:
````bash
python tools/test.py --cfg experiments/lip/seg_hrnet_w48_473x473_sgd_lr7e-3_wd5e-4_bs_40_epoch150.yaml \
                     DATASET.TEST_SET list/lip/testvalList.txt \
                     TEST.MODEL_FILE hrnet_ocr_lip_5648_torch04.pth \
                     TEST.FLIP_TEST True \
                     TEST.NUM_SAMPLES 0
````
Evaluating HRNet+OCR on the COCO-Stuff validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cocostuff/seg_hrnet_ocr_w48_520x520_ohem_sgd_lr1e-3_wd1e-4_bs_16_epoch110.yaml \
                     DATASET.TEST_SET list/cocostuff/testval.lst \
                     TEST.MODEL_FILE hrnet_ocr_cocostuff_3965_torch04.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.MULTI_SCALE True TEST.FLIP_TEST True
````
Evaluating HRNet+OCR on the ADE20K validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/ade20k/seg_hrnet_ocr_w48_520x520_ohem_sgd_lr2e-2_wd1e-4_bs_16_epoch120.yaml \
                     DATASET.TEST_SET list/ade20k/testval.lst \
                     TEST.MODEL_FILE hrnet_ocr_ade20k_4451_torch04.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.MULTI_SCALE True TEST.FLIP_TEST True
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

@article{YuanCW19,
  title={Object-Contextual Representations for Semantic Segmentation},
  author={Yuhui Yuan and Xilin Chen and Jingdong Wang},
  booktitle={ECCV},
  year={2020}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)

## Acknowledgement
We adopt sync-bn implemented by [InplaceABN](https://github.com/mapillary/inplace_abn) for PyTorch 0.4.1 experiments and the official 
sync-bn provided by PyTorch for PyTorch 1.10 experiments.

We adopt data precosessing on the PASCAL-Context dataset, implemented by [PASCAL API](https://github.com/zhanghang1989/detail-api).

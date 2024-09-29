# OrientedFormer: An End-to-End Transformer-Based Oriented Object Detector in Remote Sensing Images

The Chinese Version is below (中文版在下面).

## Introduction

The paper is officially accepted by IEEE Transactions on Geoscience and Remote Sensing (**TGRS 2024**).

TGRS paper link https://ieeexplore.ieee.org/document/10669376

If you like it, please click on star.

## Installation

Please refer to [Installation](https://mmrotate.readthedocs.io/en/1.x/get_started.html) for more detailed instruction.

**Note**: Our codes base on the newest version mmrotate-1.x, not mmrotate-0.x.

**Note**: All of our codes can be found in path './projects/OrientedFormer/'.

## Data Preparation

DOTA and DIOR-R : Please refer to [Preparation](https://github.com/open-mmlab/mmrotate/tree/1.x/tools/data) for more detailed data preparation.

ICDAR2015 : (1) Download ICDAR2015 dataset from [official link](https://rrc.cvc.uab.es/?ch=4&com=introduction). (2) The data structure is as follows:

```bash
root
├── icdar2015
│   ├── ic15_textdet_train_img
│   ├── ic15_textdet_train_gt
│   ├── ic15_textdet_test_img
│   ├── ic15_textdet_test_gt
```

## Train

1). DIOR-R

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py 2
```

2). DOTA-v1.0

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_train.sh  projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

3). DOTA-v1.5

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py 2
```

4). DOTA-v2.0

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py 2
```

5). ICDAR2015

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py 2
```

## Test

1). DIOR-R

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py 2
```

2). DOTA-v1.0

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

Upload results to DOTA official [website](https://captain-whu.github.io/DOTA/evaluation.html).

3). DOTA-v1.5

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py 2
```

Upload results to DOTA official [website](https://captain-whu.github.io/DOTA/evaluation.html).

4). DOTA-v2.0

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py 2
```

Upload results to DOTA official [website](https://captain-whu.github.io/DOTA/evaluation.html).

5). ICDAR2015

Get result submit.zip

```
python tools/test.py projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_160e_icdar2015.pth
```

 Calculate precision, recall and F-measure. The script.py adapted from [official website](https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1).

```
pip install Polygon3
python projects/icdar2015_evaluation/script.py –g=gt.zip –s=submit.zip
```

## Main Result





## Cite OrientedFormer

```
@ARTICLE{10669376,
  author={Zhao, Jiaqi and Ding, Zeyu and Zhou, Yong and Zhu, Hancheng and Du, Wen-Liang and Yao, Rui and El Saddik, Abdulmotaleb},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={OrientedFormer: An End-to-End Transformer-Based Oriented Object Detector in Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  keywords={Encoding;Object detection;Proposals;Detectors;Remote sensing;Current transformers;Position measurement;End-to-end detectors;oriented object detection;positional encoding (PE);remote sensing;transformer},
  doi={10.1109/TGRS.2024.3456240}}
```

***



# OrientedFormer: An End-to-End Transformer-Based Oriented Object Detector in Remote Sensing Images

## 简介

论文被IEEE Transactions on Geoscience and Remote Sensing (**TGRS 2024**) 接受。

TGRS官方论文链接 https://ieeexplore.ieee.org/document/10669376

如果喜欢，请点一点小星星收藏。

## 安装

参考mmrotate-1.x的官方[安装教程](https://mmrotate.readthedocs.io/en/1.x/get_started.html)获取更多安装细节。

注意：代码是基于最新版本的mmrotate-1.x，而不是旧版的mmrotate-0.x。

注意：orientedformer的代码全部在路径'./projects/OrientedFormer/'

## 数据准备

DOTA and DIOR-R : 请参考mmrotate-1.x官方[数据处理](https://github.com/open-mmlab/mmrotate/tree/1.x/tools/data)对DOTA和DIOR-R的准备方法。

ICDAR2015 : (1) 从官方网站下载 [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=introduction) 数据集。(2) 数据路径结构如下所示：

```bash
root
├── icdar2015
│   ├── ic15_textdet_train_img
│   ├── ic15_textdet_train_gt
│   ├── ic15_textdet_test_img
│   ├── ic15_textdet_test_gt
```

## 训练

1). DIOR-R

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py 2
```

2). DOTA-v1.0

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_train.sh  projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

3). DOTA-v1.5

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py 2
```

4). DOTA-v2.0

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py 2
```

5). ICDAR2015

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py 2
```

## 测试

1). DIOR-R

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py 2
```

2). DOTA-v1.0

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

将结果上传 [DOTA](https://captain-whu.github.io/DOTA/evaluation.html)官方网站。

3). DOTA-v1.5

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py 2
```

将结果上传 [DOTA](https://captain-whu.github.io/DOTA/evaluation.html)官方网站。

4). DOTA-v2.0

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py 2
```

将结果上传 [DOTA](https://captain-whu.github.io/DOTA/evaluation.html)官方网站。

5). ICDAR2015

得到结果submit.zip

```
python tools/test.py projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_160e_icdar2015.pth
```

计算precision, recall 和 F-measure

```
pip install Polygon3
python projects/icdar2015_evaluation/script.py –g=gt.zip –s=submit.zip
```

## 主要结果



## 引用 OrientedFormer

```
@ARTICLE{10669376,
  author={Zhao, Jiaqi and Ding, Zeyu and Zhou, Yong and Zhu, Hancheng and Du, Wen-Liang and Yao, Rui and El Saddik, Abdulmotaleb},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={OrientedFormer: An End-to-End Transformer-Based Oriented Object Detector in Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  keywords={Encoding;Object detection;Proposals;Detectors;Remote sensing;Current transformers;Position measurement;End-to-end detectors;oriented object detection;positional encoding (PE);remote sensing;transformer},
  doi={10.1109/TGRS.2024.3456240}}
```


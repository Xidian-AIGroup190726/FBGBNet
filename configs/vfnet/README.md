# VarifocalNet

> [VarifocalNet: An IoU-aware Dense Object Detector](https://arxiv.org/abs/2008.13367)

<!-- [ALGORITHM] -->

## Abstract

Accurately ranking the vast number of candidate detections is crucial for dense object detectors to achieve high performance. Prior work uses the classification score or a combination of classification and predicted localization scores to rank candidates. However, neither option results in a reliable ranking, thus degrading detection performance. In this paper, we propose to learn an Iou-aware Classification Score (IACS) as a joint representation of object presence confidence and localization accuracy. We show that dense object detectors can achieve a more accurate ranking of candidate detections based on the IACS. We design a new loss function, named Varifocal Loss, to train a dense object detector to predict the IACS, and propose a new star-shaped bounding box feature representation for IACS prediction and bounding box refinement. Combining these two new components and a bounding box refinement branch, we build an IoU-aware dense object detector based on the FCOS+ATSS architecture, that we call VarifocalNet or VFNet for short. Extensive experiments on MS COCO show that our VFNet consistently surpasses the strong baseline by ∼2.0 AP with different backbones. Our best model VFNet-X-1200 with Res2Net-101-DCN achieves a single-model single-scale AP of 55.1 on COCO test-dev, which is state-of-the-art among various object detectors.

<div align=center>
<img src="https://user-images.githubusercontent.com/9102141/97464778-4b9ab000-197c-11eb-9283-ab2907ee0252.png"/>
</div>

## Introduction

**VarifocalNet (VFNet)** learns to predict the IoU-aware classification score which mixes the object presence confidence and localization accuracy together as the detection score for a bounding box. The learning is supervised by the proposed Varifocal Loss (VFL), based on a new star-shaped bounding box feature representation (the features at nine yellow sampling points). Given the new representation, the object localization accuracy is further improved by refining the initially regressed bounding box. The full paper is available at: [https://arxiv.org/abs/2008.13367](https://arxiv.org/abs/2008.13367).



## Citation

```latex
@article{zhang2020varifocalnet,
  title={VarifocalNet: An IoU-aware Dense Object Detector},
  author={Zhang, Haoyang and Wang, Ying and Dayoub, Feras and S{\"u}nderhauf, Niko},
  journal={arXiv preprint arXiv:2008.13367},
  year={2020}
}
```

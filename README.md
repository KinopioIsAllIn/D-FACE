# From Pixels to Semantics: Unified Facial Action Representation Learning for Micro-Expression Analysis

## Introduction

Official Pytorch Implementation of 'From Pixels to Semantics: Unified Facial Action Representation Learning for Micro-Expression Analysis' (ICLR2026)

The current version is still to be updated.

## Data preparation
We can not provide the dataset, please apply for each dataset by yourself.
After downloading the files, move them into the ```datasets/``` directory.

## Train and validation on CAS(ME)^3 dataset for 4-class recognition
``` bash
python train_casme3_4c.py
```

## Citation

```
@inproceedings{deng2026from,
  title={From Pixels to Semantics: Unified Facial Action Representation Learning for Micro-Expression Analysis},
  author={Yicheng Deng and Hideaki Hayashi and Hajime Nagahara},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}}
```

## Acknowlegment
We implement our source code based on [LAPA](https://github.com/LatentActionPretraining/LAPA) and [Exp-CLIP](https://github.com/zengqunzhao/Exp-CLIP).

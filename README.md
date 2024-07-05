# Intriguing Properties of Hyperbolic Embeddings in Vision-Language Models
[Sarah Ibrahimi](https://saibr.github.io), [Mina Ghadimi Atigh](https://minaghadimi.github.io), [Nanne van Noord](https://nanne.github.io), [Pascal Mettes](https://staff.fnwi.uva.nl/p.s.m.mettes/), [Marcel Worring](https://staff.fnwi.uva.nl/m.worring/)

**TMLR 2024** [[`Paper`](https://openreview.net/forum?id=P5D2gfi4Gg)] 

## Overview

This repository contains details on reproducing the results from our TMLR 2024 paper. Our paper contains results on 7 benchmarks, in this repository we provide the code and steps to reproduce the results for one specific benchmark.

For the experiments in this paper, we used the following 7 codebases:

### Spatial Reasoning

[VL-Checklist](https://github.com/om-ai-lab/VL-CheckList)

[VG-Relations](https://github.com/mertyg/vision-language-models-are-bows)

[CLIPbind-r](https://github.com/marthaflinderslewis/clip-binding)

### Ambiguity Resolution

[Visual Word Sense Disambiguation (VWSD)](https://github.com/asahi417/visual-wsd-baseline)

[Propaganda Memes](https://github.com/di-dimitrov/propaganda-techniques-in-memes?tab=readme-ov-file) - This repository is only used for the dataset. A dataloader and evaluation code are created separately for this task and integrated in the code for VWSD.

[My Reaction When](https://github.com/yalesong/pvse?tab=readme-ov-file#mrw) - This repository is only used for the dataset. A dataloader and evaluation code are created separately for this task and integrated in the code for VWSD.

### Out-of-distribution Detection

[OpenOOD](https://github.com/Jingkang50/OpenOOD)

## General comments

For each task, we integrate the code from Desai et al. in [Hyperbolic Image-Text Representations](https://arxiv.org/pdf/2304.09172) to evaluate the Euclidean and hyperbolic model (MERU). This code can be found [here](https://github.com/facebookresearch/meru/tree/main). 

In our repository, we show for one of the benchmarks for Spatial Reasoning, *VG-Relations*, how to integrate the hyperbolic models. We took the original repository of *VG-Relations* as a starting point and thank the creators for releasing this code. An overview of this integration process should help in integrating hyperbolic models easily in any other existing codebase that has an integration of the [CLIP model](https://github.com/openai/CLIP).

**Note that this repository is mainly a merge of [VG-Relations](https://github.com/mertyg/vision-language-models-are-bows) and [MERU](https://github.com/facebookresearch/meru/tree/main), with a few additional modifications. We do not claim ownership for the code from their repositories.**

## Installation

We recommend using a conda environment for each task. The results in the original MERU paper could be reproduced by version 1.11.0 of PyTorch and CUDA 10.2, therefore we used it for all our experiments and installed it by the following command:

```
pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
```

We made an export of the conda environment that we used for the VG-Relations benchmark. However, we recommend creating the conda environment by installing the packages one by one, starting from the most important packages such as timm, clip, hydra-core, omegaconf, opencv-python etc., to make sure you end up with a compatible collection of packages. Simply ```conda env create``` from the yml file will most likely not work.

## Trained model checkpoints

The trained model checkpoints for MERU and CLIP for small, base and large ViT models can be found [here](https://github.com/facebookresearch/meru/tree/main). 

## Integration of the hyperbolic model (MERU)

To get a better understanding of the process, the ```class ZeroShotClassificationEvaluator``` in the [MERU code](https://github.com/facebookresearch/meru/blob/6d4430055f58354539f68668175135256f99d47f/meru/evaluation/classification.py) is a good starting point.

To make integration easy, we create one folder named **meru** with all the files that can be used directly from the MERU repository without any modifications. We create ```meru/config.py``` with a function to simplify loading the models.

Besides, we have to modify the following files:

```model_zoo/clip_models.py``` - each codebase usually has a class for the CLIP model, we create a new class for the hyperbolic clip model. This is done based on the original MERU code. For all our experiments, also for the Euclidean CLIP trained on the RedCaps dataset, we use this new class.

We load the model in the get_model function in ```model_zoo/__init__.py```. For this we use the ```prepare_model``` function from ```meru/config.py```, the ```CheckpointManager``` provided by MERU and the hyperbolic ```CLIPWrapper```.

We write a new function to evaluate the scores based on a bootstrapping technique. This can be found in ```dataset_zoo/aro_datasets.py```

Besides these steps, we write a new main file, in this case ```main_hyperbolic.py```. 

Note that these are the steps to integrate MERU in an existing codebase for zero-shot evaluation. More changes are needed for finetuning.

## Evaluation

Make sure to set ```download=True``` in the following line in ```main_hyperbolic.py``` to download the dataset if it is not there yet.
```
vgr_dataset = VG_Relation(image_preprocess=preprocess, download=True, root_dir=root_dir)
```

The evaluation step can be done by the following command:
```
python main_hyperbolic.py --root ROOT_DIR --model_name MODEL_NAME
```

We assume the following structure:
```
/home/user/projects/hyperbolic-embeddings-vl-models/
/home/user/datasets
/home/user/pretrained_models
```

In this case, ROOT_DIR will be ```/home/user```. Using a different structure requires modifying some parts in the code for model or dataloading.

## Citation
If you use our code, please consider citing our paper:

```
@article{
  ibrahimi2024tmlr,
  title={Intriguing Properties of Hyperbolic Embeddings in Vision-Language Models},
  author={Sarah Ibrahimi and Mina Ghadimi Atigh and Nanne van Noord and Pascal Mettes and Marcel Worring},
  journal={TMLR},
  year={2024},
  url={https://openreview.net/forum?id=P5D2gfi4Gg}
}
```


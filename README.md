# Weakly Supervised Pre-Training for Multi-Instance Learning on Whole Slide Pathological Image

Pytorch implementation of weakly supervised pre-training framework we proposed.

## Installation

Install [anaconda/miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

Required packages
```
$ conda env create --file simmil_environment.yml
$ conda activate simmil
```

## Stage1: Data Pre-processing

Download NCTCRC from [ZENODO](https://zenodo.org/records/1214456)

Create NCTCRC-BAGS
```python
python pretrain/gen_bag_NCTCRC.py
```
Download TCGA-NSCLC and TCGA-BRCA using [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/), and do the segmentation using [CLAM](https://github.com/mahmoodlab/CLAM)

## Stage2: Pretrain a Feature Extractor
```python
CUDA_VISIBLE_DEVISES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --dataset [C16/TCGA/BRCA] --loss sce --arch [resnet18/resnet50] 
```

## Stage3: Compute Instance Features
For backbone ResNet18, ResNet50, CTransPath
```
python aggregation/downstream/compute_feats.py --dataset [C16/tcga] --backbone [resnet18/resnet50/ctrans]
```
For backbone HIPT
```
python pretrain/compute_HIPT_features.py
```

## Stage4: Train Aggregation Network
For NRCTRC in preliminary
```
python aggregation/preliminary/main.py 
```
For experiments of other datasets:

ABMIL, DSMIL, CLAM-SB

```
python aggregation/downstream/train.py --num_classes [according to your dataset] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features] --model [abmil/dsmil/clam]
```

TransMIL
```
python aggregation/downstream/train_transmil.py --num_classes [according to your dataset] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features] --model transmil
```

DTFD-MIL
```
python aggregation/downstream/train_DTFD.py --num_classes [according to your dataset] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features] --model DTFD
```
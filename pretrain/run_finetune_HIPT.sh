#!/bin/bash 

CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
    /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
    --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/HIPT_TCGA_epoch5_ce \
    --dataset TCGA --epochs 5 --arch HIPT --cosine True \

CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
    /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
    --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/HIPT_BRCA_epoch5_ce \
    --dataset BRCA --epochs 5 --arch HIPT --cosine True \
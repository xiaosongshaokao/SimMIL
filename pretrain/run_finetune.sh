#!/bin/bash 

# CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
#     /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
#     --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/ctrans_C16_epoch10 \
#     --dataset C16 --loss sce --epochs 10 --arch CTrans --cosine True \
#     --pretrained /remote-home/share/GraphMIL/backbone_check/debug/CTrans/ctranspath.pth \
#     --bag_label_dir /remote-home/kongyijian/MIL/SimMIL/data/C16

CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
    /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
    --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/ctrans_TCGA_epoch10 \
    --dataset TCGA --loss sce --epochs 10 --arch CTrans --cosine True \
    --pretrained /remote-home/share/GraphMIL/backbone_check/debug/CTrans/ctranspath.pth \
    --bag_label_dir /remote-home/kongyijian/MIL/SimMIL/data/TCGA

CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
    /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
    --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/HIPT_TCGA_epoch10 \
    --dataset TCGA --loss sce --epochs 10 --arch HIPT --cosine True \

CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
    /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
    --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/HIPT_BRCA_epoch10 \
    --dataset BRCA --loss sce --epochs 10 --arch HIPT --cosine True \

CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
    /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
    --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/ctrans_C16_epoch1 \
    --dataset C16 --loss sce --epochs 1 --arch CTrans --cosine True \
    --pretrained /remote-home/share/GraphMIL/backbone_check/debug/CTrans/ctranspath.pth \
    --bag_label_dir /remote-home/kongyijian/MIL/SimMIL/data/C16

CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
    /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
    --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/ctrans_TCGA_epoch1 \
    --dataset TCGA --loss sce --epochs 1 --arch CTrans --cosine True \
    --pretrained /remote-home/share/GraphMIL/backbone_check/debug/CTrans/ctranspath.pth \
    --bag_label_dir /remote-home/kongyijian/MIL/SimMIL/data/TCGA

CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
    /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
    --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/HIPT_TCGA_epoch1 \
    --dataset TCGA --loss sce --epochs 1 --arch HIPT --cosine True \

CUDA_VISIBLE_DEVICES=1,2,6,7 /root/miniconda/envs/simmil/bin/torchrun --standalone --nproc_per_node=4 \
    /remote-home/kongyijian/GraphMIL/backbone_check/main.py \
    --log_dir /remote-home/share/GraphMIL/backbone_check/debug/finetune/HIPT_BRCA_epoch1 \
    --dataset BRCA --loss sce --epochs 1 --arch HIPT --cosine True \
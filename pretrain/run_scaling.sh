#!/bin/bash 
# CUDA_VISIBLE_DEVICES=3,5,6,7 /usr/local/miniconda3/envs/mil/bin/python -m torch.distributed.launch --nproc_per_node 4 --use_env --master_port 29400 \
# /remote-home/kongyijian/GraphMIL/backbone_check/main.py --log_dir /remote-home/share/GraphMIL/backbone_check/debug/scaling/C16 --loss sce --epochs 100 \
# --dataset C16 --bag_label_dir /remote-home/kongyijian/MIL/SimMIL/data/sample10_dataset/C16/ --schedule 60 80
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/local/miniconda3/envs/mil/bin/python -m torch.distributed.launch --nproc_per_node 4 --use_env --master_port 29400 \
/remote-home/kongyijian/GraphMIL/backbone_check/main.py --log_dir /remote-home/share/GraphMIL/backbone_check/debug/scaling/TCGA --loss sce --epochs 100 \
--dataset TCGA --bag_label_dir /remote-home/kongyijian/MIL/SimMIL/data/sample10_dataset/TCGA/ --schedule 60 80
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/local/miniconda3/envs/mil/bin/python -m torch.distributed.launch --nproc_per_node 4 --use_env --master_port 29400 \
/remote-home/kongyijian/GraphMIL/backbone_check/main.py --log_dir /remote-home/share/GraphMIL/backbone_check/debug/scaling/BRCA --loss sce --epochs 100 \
--dataset TCGA --bag_label_dir /remote-home/kongyijian/MIL/SimMIL/data/sample10_dataset/BRCA/ --schedule 60 80
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/local/miniconda3/envs/mil/bin/python -m torch.distributed.launch --nproc_per_node 4 --use_env --master_port 29400 \
/remote-home/kongyijian/GraphMIL/backbone_check/main.py --log_dir /remote-home/share/GraphMIL/backbone_check/debug/scaling/TCGA_C16_BRCA --loss sce --epochs 100 \
--dataset TCGA --bag_label_dir /remote-home/kongyijian/MIL/SimMIL/data/sample10_dataset/TCGA_C16_BRCA/ --schedule 60 80 --class_num 6
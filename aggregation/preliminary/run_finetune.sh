#!/bin/bash 
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29500 \
/remote-home/kongyijian/GraphMIL/backbone_check/finetune_NCTCRC.py \
--pretrained_weights /remote-home/share/GraphMIL/backbone_check/debug/NCTCRC_WSL_ori/feature_extract_mt.pth \
--output_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/finetune_WSL_mt

# CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29500 \
# /remote-home/kongyijian/GraphMIL/backbone_check/finetune_NCTCRC.py \
# --pretrained_weights /remote-home/share/GraphMIL/backbone_check/debug/ConCL/moco_v2.pth \
# --output_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/finetune_MOCO

# CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29500 \
# /remote-home/kongyijian/GraphMIL/backbone_check/finetune_NCTCRC.py \
# --pretrained_weights /remote-home/share/GraphMIL/backbone_check/debug/ConCL/simclr.pth \
# --output_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/finetune_SIMCLR


# CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29500 \
# /remote-home/kongyijian/GraphMIL/backbone_check/finetune_NCTCRC.py --imagenet_pretrain True \
# --output_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/finetune_IP

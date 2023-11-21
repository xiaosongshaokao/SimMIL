#!/bin/bash 
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python /remote-home/kongyijian/GraphMIL/backbone_aggregation/main.py \
--log_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/IP_MAX --aggregator max_pooling --pretrained_type simpleMIL --imgnet_pretrained 
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python /remote-home/kongyijian/GraphMIL/backbone_aggregation/main.py \
--log_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/IP_MEAN --aggregator mean_pooling --pretrained_type simpleMIL --imgnet_pretrained 
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python /remote-home/kongyijian/GraphMIL/backbone_aggregation/main.py \
--log_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/WSL_MAX --aggregator max_pooling \
--load /remote-home/share/GraphMIL/backbone_check/debug/NCTCRC_WSL/feature_extract.pth --pretrained_type simpleMIL 
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python /remote-home/kongyijian/GraphMIL/backbone_aggregation/main.py \
--log_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/WSL_MEAN --aggregator mean_pooling \
--load /remote-home/share/GraphMIL/backbone_check/debug/NCTCRC_WSL/feature_extract.pth --pretrained_type simpleMIL 
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python /remote-home/kongyijian/GraphMIL/backbone_aggregation/main.py \
--log_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/MOCO_MAX --aggregator max_pooling \
--load /remote-home/share/GraphMIL/backbone_check/debug/ConCL/moco_v2.pth --pretrained_type simpleMIL
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python /remote-home/kongyijian/GraphMIL/backbone_aggregation/main.py \
--log_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/MOCO_MEAN --aggregator mean_pooling \
--load /remote-home/share/GraphMIL/backbone_check/debug/ConCL/moco_v2.pth --pretrained_type simpleMIL
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python /remote-home/kongyijian/GraphMIL/backbone_aggregation/main.py \
--log_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/SIMCLR_MAX --aggregator max_pooling \
--load /remote-home/share/GraphMIL/backbone_check/debug/ConCL/simclr.pth --pretrained_type simpleMIL
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/miniconda3/envs/mil/bin/python /remote-home/kongyijian/GraphMIL/backbone_aggregation/main.py \
--log_dir /remote-home/kongyijian/GraphMIL/backbone_aggregation/debug/SIMCLR_MEAN --aggregator mean_pooling \
--load /remote-home/share/GraphMIL/backbone_check/debug/ConCL/simclr.pth --pretrained_type simpleMIL
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import os, sys
from datasets.my_imagefolder import MyImageFolder
from datasets.camelyon17 import CameLyon17
from datasets.debug_dataset import DebugDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from runners import LinClsRunner, MoCoRunner, MeanTeacherRunner, OracleRunner
from configs import LinClsParser, MoCoParser, MeanTeacherParser 
from utils.misc import get_timestamp
from utils.path import mkdir_or_exist

# os.environ["CUDA_VISIBLE_DEVICES"] =  '1, 2, 3, 4'
runner_dict = dict(lincls=LinClsRunner,
                   moco=MoCoRunner,
                   mt=MeanTeacherRunner,
                   oracle=OracleRunner,
)
parser_dict = dict(lincls=LinClsParser,
                   moco=MoCoParser,
                   mt=MeanTeacherParser,
                   oracle=LinClsParser)
def parse_args():
    parser = argparse.ArgumentParser('Backbone Experiment argument parser')
    parser.add_argument('--runner_type', type=str, default='lincls')

    return parser.parse_known_args()

def main():
    known_args, args = parse_args()
    runner_type = known_args.runner_type
    args = parser_dict[runner_type]().parse_args(args)
    args.timestamp = get_timestamp()
    mkdir_or_exist(args.log_dir)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('ngpu', ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, runner_type))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, runner_type)

def main_worker(gpu, ngpus_per_node, args, runner_type):
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        args.master = True
    else:
        args.master = False
    runner = runner_dict[runner_type](gpu, ngpus_per_node, args)
    runner.run()


if __name__=="__main__":
    mp.set_start_method('forkserver')
    main()
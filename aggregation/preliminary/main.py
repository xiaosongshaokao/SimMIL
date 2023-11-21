import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import argparse
import random
import builtins
import shutil
import time
import warnings
import os, sys
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
from utils.misc import get_timestamp
from utils.path import mkdir_or_exist
from runners.base_runner import BaseRunner
from runners.ins_runner import InsRunner
from runners.assumption_runner import AssumptionRunner

from configs.base import BaseParser
from configs.ins_mil import InsMILParser
from configs.assumption import AssumptionParser



runner_dict = dict(base=BaseRunner,
                   ins=InsRunner,
                   assumption=AssumptionRunner
)
parser_dict = dict(base=BaseParser,
                   ins=InsMILParser,
                   assumption=AssumptionParser)
def parse_args():
    parser = argparse.ArgumentParser('Backbone Experiment argument parser')
    parser.add_argument('--runner_type', type=str, default='assumption')

    return parser.parse_known_args()

def main():
    known_args, args = parse_args()
    runner_type = known_args.runner_type
    args = parser_dict[runner_type]().parse_args(args)
    args.timestamp = get_timestamp()
    mkdir_or_exist(args.log_dir)
    runner = runner_dict[runner_type](args)
    runner.run()

if __name__=="__main__":
    print(1)
    main()
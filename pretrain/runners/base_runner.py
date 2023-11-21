import argparse
import builtins
import os
import random
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
import numpy as np
from utils.logger import Logger
import json

class BaseRunner(object):
    def __init__(self, gpu, ngpus_per_node, args):
        
        args.gpu = gpu
        self.gpu = gpu
        self.ngpus_per_node = ngpus_per_node
        self.args = args
        self.logger = Logger(self.args)
        # self.init_ddp()
    
    def run(self):
        raise NotImplementedError

    def init_ddp(self):
        # suppress printing and auto backup if not master
        if self.args.multiprocessing_distributed and self.gpu != 0:
            def print_pass(*args):
                pass

            def auto_backup(root='./'):
                pass

            self.logger.info = print_pass
            self.logger.auto_backup = auto_backup
        self.logger.init_info()
        # self.logger.auto_backup()
        self.logger.info("Configuration:\n{}".format(json.dumps(self.args.__dict__, indent=4)))
        if self.gpu is not None:
            self.logger.info("Use GPU: {} for training".format(self.gpu))
        if self.args.distributed:
            if self.args.dist_url == "env://" and self.args.rank == -1:
                self.args.rank = int(os.environ["RANK"])
            
            if self.gpu is not None:
                self.set_ddp_batch_size()
                self.set_ddp_num_workers()

            if self.args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.args.rank = -1 * (self.args.rank * self.ngpus_per_node + self.gpu + 1)
                print('rank', self.args.rank)
            
            import datetime

            # 创建一个表示1分钟时间间隔的timedelta对象
            one_minute = datetime.timedelta(minutes=1)
            dist.init_process_group(backend=self.args.dist_backend, init_method='env://',
                                    world_size=self.args.world_size, rank=self.args.rank, timeout=one_minute)
            

    def set_ddp_batch_size(self):
        self.args.batch_size = int(self.args.batch_size / self.ngpus_per_node)

    def set_ddp_num_workers(self):
        self.args.workers = int((self.args.workers + self.ngpus_per_node - 1) / self.ngpus_per_node)

    def set_device(self, model):
        if self.args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                model.cuda(self.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu)
            model = model.cuda(self.args.gpu)
        else:
            #将模型放到所有可用的 GPU 上
            if torch.cuda.is_available():
                model = nn.DataParallel(model)

        return model
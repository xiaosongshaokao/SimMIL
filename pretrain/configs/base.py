import argparse
import os
import torch
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append('CTrans')
model_names.append('HIPT')

class BaseParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='backbone checking base args')
        self.parser.add_argument('--data', metavar='DIR',default='/remote-home/kongyijian/MIL/SimMIL/data/NCTCRC',
                    help='path to dataset')
        self.parser.add_argument('--log_dir', type=str, default='/remote-home/share/GraphMIL/backbone_check/debug/TCGA_C16_BRCA/')
        self.parser.add_argument('--dataset', type=str, default='TCGA', choices=['CAMELYON16', 'CIFAR10' ,'MNIST','CAMELYON17', 'NCTCRC', 'NCTCRC-BAGS', 'TCGA', 'C16', 'BRCA'])

        self.parser.add_argument('--train-subdir', type=str, default='NCT-CRC-HE-100K',
                            help='the subdirectory inside the data directory that contains the training data')
        self.parser.add_argument('--eval-subdir', type=str, default='CRC-VAL-HE-7K',
                            help='the subdirectory inside the data directory that contains the evaluation data')
        self.parser.add_argument('--loss', type=str, default='CE',
                            help='loss type, CE, Focal, seesaw, LDAMLoss, sce')
                            
        self.parser.add_argument('-b', '--batch-size', default=64, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total'
                                'batch size of all GPUs on the current node when '
                                'using Data Parallel or Distributed Data Parallel')
        self.parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        self.parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        self.parser.add_argument('--no_norm', action='store_true', default=False ,help='Omitted if not using NCTCRC dataset')
        self.parser.add_argument('--imgnet_pretrained', default=None ,action='store_true', help='Whether use imagnet pretrain')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')# 
        self.parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                            choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet50)')
        self.parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
        self.parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                            help='number of data loading workers (default: 32)')
        self.parser.add_argument('--class_num', '-cn', type=int, default=2, 
                            help='class number') 
        self.parser.add_argument('--cosine', type=bool, default=False, 
                            help='use cosine lr schedule or not')
        self.parser.add_argument("--save_result", action='store_true',
                            help='save evaluation result')
        self.parser.add_argument("--save_result_name", type=str, default="evaluate_result.pth",
                            help='file name of saved evaluation result. Only useful when save_result is True.')
        self.parser.add_argument('--debug', action='store_true', help='debug use')
        self.parser.add_argument('-p', '--print-freq', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        # distributed learning / multi-processing part
        self.parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        self.parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        self.parser.add_argument('--dist-url', default='env:', type=str,
                            help='url used to set up distributed training')
        self.parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        self.parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        self.parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        self.parser.add_argument('--multiprocessing-distributed', default = False, action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
    

        self.parser.add_argument('--no_fixed_trunk', default=True, type=bool,
                            help='freeze all layers but the last fc or not')
        self.parser.add_argument('--pretrained', default=None, type=str,
                            help='whether load from pre-trained')
        self.parser.add_argument('--load', default=False, type=bool,
                            help='whether load to resume')
        self.parser.add_argument('--sample_file', default='/remote-home/kongyijian/GraphMIL/samples/nctcrc/49995_balanced_labels/08.txt',
                            help='moco sample file')
        # self.parser.add_argument('--evaluate_before_train', default=False, type=bool,
        #                     help='whether evaluate before train')
        # self.parser.add_argument('--validate_interval', default=5, type=int,
        #                     help='validate interval')
        # self.parser.add_argument('--eval_class_num', default=2, type=int,
        #                     help='validate interval')
        

    def parse_args(self, args=None):
        return self.parser.parse_args() if args is None else self.parser.parse_args(args)
        
import argparse
import os
import torch
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class BaseParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='backbone checking base args')
        self.parser.add_argument('--data',default='' ,metavar='DIR',
                    help='path to dataset')
        self.parser.add_argument('--log_dir', type=str, default='aggregation/preliminary/experiments/')
        self.parser.add_argument('--train-subdir', type=str, default='NCT-CRC-HE-100K',
                            help='the subdirectory inside the data directory that contains the training data')
        self.parser.add_argument('--eval-subdir', type=str, default='CRC-VAL-HE-7K',
                            help='the subdirectory inside the data directory that contains the evaluation data')
        self.parser.add_argument('--train_label_file', type=str, 
                            default='/remote-home/kongyijian/MIL/SimMIL/data/samples/nctcrc_bags_std/BL50/target8/train_ins.txt',
                            help='Label file for training')
        self.parser.add_argument('--val_label_file', type=str, 
                            default='/remote-home/kongyijian/MIL/SimMIL/data/samples/nctcrc_bags_std/BL50/target8/val_ins.txt',
                            help='Label file for testing')
        self.parser.add_argument('--aggregator', type=str, default='max_pooling', choices=['mean_pooling', 
                                                                                            'max_pooling', 
                                                                                            'attention', 
                                                                                            'gcn', 'abmil', 'dsmil', 'transmil'])
        self.parser.add_argument('-b', '--batch_size', default=8, type=int,
                            metavar='N',
                            help="mini-batch-size, how much bags in a batch")
        self.parser.add_argument('--epochs', default=50, type=int, metavar='N',
                            help='number of total epochs to run')
        self.parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        self.parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                            choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet50)')
        self.parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
        self.parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 32)')
        self.parser.add_argument('--class_num', '-cn', type=int, default=2, 
                            help='class number')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        self.parser.add_argument('--evaluate_before_train', '-ebt', action='store_true',
                            help='evaluate model before training')
        self.parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                            metavar='W', help='weight decay (default: 0.)',
                            dest='weight_decay')
        self.parser.add_argument('--debug', action='store_true', help='debug use')
        self.parser.add_argument('-p', '--print-freq', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        # distributed learning / multi-processing part
        self.parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        self.parser.add_argument('--load', default='', type=str, metavar='PATH',
                            help='load state dict from pretrained')

        self.parser.add_argument('--pretrained_type', '-pt', default='simpleMIL', type=str,
                                choices=['moco', 'oracle', 'mt', 'simpleMIL'],
                                help='pretrained model type')
        self.parser.add_argument('--imgnet_pretrained' ,action='store_true', help='Whether use imagnet pretrain')
        self.parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        self.parser.add_argument('--schedule', default=[30, 40], nargs='*', type=int,
                            help='learning rate schedule (when to drop lr by a ratio)')
        self.parser.add_argument('--validate_interval', type=int, default=10, 
                            help='validation interval')
        self.parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')

    def parse_args(self, args=None):
        return self.parser.parse_args() if args is None else self.parser.parse_args(args)
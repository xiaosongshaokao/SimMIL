# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from DINO library:
https://github.com/facebookresearch/dino
"""


import os
import argparse
import json
import copy
import itertools
import random
import torch
import torch.backends.cudnn as cudnn
import runners.utils as utils
from tqdm import tqdm
import torch.distributed as dist
from torchvision import models as torchvision_models
import torchvision.transforms as transforms
from pathlib import Path
from torch import nn
from torchvision import transforms as pth_transforms
from PIL import Image
from datasets.my_imagefolder_eval import MyImageFolder
# from loader import ImageFolder
import pickle
# import redis
from collections import defaultdict
import torch.nn.functional as F
from PIL import ImageFilter
from sklearn.metrics import classification_report, roc_auc_score, classification_report, confusion_matrix

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def build_dataset(augmentation, mode='train'):
        """
        Build dataset object based on self.args.
        """
        # logger.info("Building dataset")
        no_norm = False
        data = '/remote-home/kongyijian/MIL/SimMIL/data/NCTCRC'
        eval_subdir = 'CRC-VAL-HE-7K'
        bag_label_dir = '/remote-home/kongyijian/MIL/SimMIL/data/samples/nctcrc_bags_std/BL50/target8'
        if mode=='train':
            if no_norm:
                train_dir = os.path.join(data, 'NCT-CRC-HE-100K-NONORM')
            else:
                train_dir = os.path.join(data, 'NCT-CRC-HE-100K')
        else:
            train_dir = os.path.join(data, eval_subdir)

        dataset = MyImageFolder(train_dir, augmentation)

        postfix = 'train.txt' if mode=='train' else 'val.txt'
        bag_label_file = os.path.join(bag_label_dir, postfix)
        with open(bag_label_file, 'r') as f:
            samples = [x.strip().split(" ") for x in f.readlines()]
            print("=> Using MIL Setting")
            samples = list(map(lambda x:(x[0], int(x[1])), samples))
            dataset.labels = [int(sample[1]) for sample in samples]
            dataset.samples = samples
            print("Examples after MIL: {}".format(dataset.samples[:5]))


        return dataset

def eval_linear(args):
    utils.init_distributed_mode(args)
    args.lock_rank = int(os.environ["LOCAL_RANK"])
    # print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # fix the seed for reproducibility 
    utils.fix_random_seeds(args.seed)
    
    # ============ preparing data ... ============
    if args.arch == 'dalle_encoder':
        train_transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(112),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
        ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(128, interpolation=3),
            pth_transforms.CenterCrop(112),
            pth_transforms.ToTensor(),
        ])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    dataset_train = build_dataset(train_transform, "train")
    dataset_val = build_dataset(val_transform, "validation")
    train_sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu * 4,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu * 4,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = torchvision_models.__dict__[args.arch]()
    for name, param in model.named_parameters():
        param.requires_grad = False
    embed_dim = 512
    model.fc = nn.Identity()
    model.cuda()
    model.eval()

    print(f"Model built.")
    # load weights to evaluate
    # utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    state_dict = torch.load(args.pretrained_weights, map_location="cpu")['state_dict']
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))

    # args.lrs = [base*10 for base in  [10**k for k in range(-5, 1)]]
    args.lrs = [0.1]
    if not args.sweep_lr_only:
        args.wds = [0, 1e-6]
        args.optims = ['sgd', 'lars']
    else:
        args.wds = [0]
        args.optims = ['sgd']
    args.permutes = list(itertools.product(args.lrs, args.wds, args.optims))

    linear_classifiers = nn.ModuleList()
    optimizers = []
    schedulers = []
    
    for pm in args.permutes:
        lr, wd, optim = pm
        linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
        linear_classifier.cuda()
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.lock_rank])
        linear_classifiers.append(linear_classifier)

        # set optimizer
        parameters = linear_classifier.parameters()
        optimizer = torch.optim.SGD if optim == 'sgd' else utils.LARS
        optimizer = optimizer(
            parameters,
            lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256, # linear scaling rule
            momentum=0.9,
            weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
        
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    # Optionally resume from a checkpoint

    to_restore = {"epoch": 0, "best_acc": 0, "best_auc": 0.}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            state_dict=linear_classifiers)
        for optimizer, scheduler in zip(optimizers, schedulers):
            utils.restart_from_checkpoint(
                os.path.join(args.output_dir, args.load_from),
                optimizer=optimizer,
                scheduler=scheduler)
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    best_auc = to_restore["best_auc"]
    best_acc_sweep_lr_only = to_restore["best_acc"]
    target = torch.LongTensor(dataset_val.labels)

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train(model, linear_classifiers, optimizers, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, args.permutes, args.lock_rank)
        for scheduler in schedulers:
            scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats, buffer_dict = validate_network(val_loader, model, linear_classifiers, args.permutes, args.lock_rank)
            
            group_best_acc = 0
            group_best_acc_hidx = 0
            group_best_acc_sweep_lr_only = 0
            group_best_auc=0
            group_best_auc_hidx = 0
            # auc_list=[]
            for group, pm in enumerate(args.permutes):
                lr, wd, optim = pm

                auc_score = roc_auc_score(target, torch.cat(buffer_dict["pred_labels{}".format(group)]))*100
                # auc_list.append(auc_score)
                if auc_score >= group_best_auc:
                    group_best_auc_hidx = group
                    group_best_auc = auc_score
                # print(f"Accuracy at epoch {epoch} with lr {lr:.5f} wd {wd:.0e} optim {optim:4} of the network \
                #         on the {len(dataset_val)} test images: {test_stats['acc{}@1'.format(group)]:.1f}%")
                if group % (len(args.wds) * len(args.optims)) == 0:
                    group_best_acc_sweep_lr_only = max(group_best_acc_sweep_lr_only, test_stats['acc{}@1'.format(group)])
                # group_best_acc = max(group_best_acc, test_stats['acc{}@1'.format(group)])
                if test_stats['acc{}@1'.format(group)] >= group_best_acc:
                    group_best_acc_hidx = group
                    group_best_acc = test_stats['acc{}@1'.format(group)]
                print('* [Lr {lr:.5f} Wd {wd:.0e} Optim {optim:4}] Acc@1 {top1:.3f} AUC {AUC:.3f} loss {losses:.3f}'
            .format(lr=lr, wd=wd, optim=optim, 
                    top1=test_stats['acc{}@1'.format(group)], 
                    AUC=auc_score,
                    losses=test_stats['loss{}'.format(group)]))

            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                        #  }
                        #  **{k: v for k, v in buffer_dict.items()}}
                        **{'Max accuracy so far': best_acc,
                         'Max AUC so far': best_auc,
                         'Max accuracy with sweeping lr only so far':best_acc_sweep_lr_only}}
            

            if utils.is_main_process() and (group_best_acc >= best_acc):
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifiers.state_dict(),
                    "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                    "best_acc": group_best_acc,
                    'best_acc_hidx': group_best_acc_hidx,
                    "best_auc": group_best_auc,
                    'best_auc_hidx': group_best_auc_hidx,
                    "best_acc_sweep_lr_only": group_best_acc_sweep_lr_only,
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_{}_linear.pth".format(args.checkpoint_key)))

            best_acc = max(best_acc, group_best_acc)
            best_auc = max(best_auc, group_best_auc)
            best_acc_sweep_lr_only = max(best_acc_sweep_lr_only, group_best_acc_sweep_lr_only)
            print(f'Max accuracy so far: {best_acc:.2f}%')
            print(f'Max AUC so far: {best_auc:.4f}%')
            print(f'Max accuracy with sweeping lr only so far: {best_acc_sweep_lr_only:.2f}%')

    lr, wd, optim = args.permutes[group_best_acc_hidx]
    print("Training of the supervised linear classifier on frozen features completed.\n",
              "Top-1 test accuracy: {acc:.1f}\n".format(acc=best_acc),
              "Top-1 test accuracy with sweeping lr only: {acc:.1f}\n".format(acc=best_acc_sweep_lr_only),
              "Optim configs with top-1 test accuracy: lr {lr:.5f}, wd {wd:.0e}, optim {optim:4}\n".format(lr=lr, wd=wd, optim=optim))


def train(model, linear_classifiers, optimizers, loader, epoch, n, avgpool, permutes, lock_rank):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    metric_logger = utils.MetricLogger(delimiter="  ")
    for group, _ in enumerate(permutes):
        metric_logger.add_meter('lr{}'.format(group), utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for data, label in metric_logger.log_every(tqdm(loader), 389, header):
        # move to gpu
        inp = data.cuda()
        target = label.cuda()

        # forward
        with torch.no_grad():
            output = model(inp)
        for param in model.parameters():
            param.grad = None
      
        losses = []
        for linear_classifier, optimizer in zip(linear_classifiers, optimizers):
            linear_classifier.train()
            # for param in linear_classifier.parameters():
            #     param.requires_grad = True
            pred = linear_classifier(output)

            # compute cross entropy loss
            loss = nn.CrossEntropyLoss()(pred, target)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()

            losses.append(loss)

        # log 
        torch.cuda.synchronize()
        for group, (loss, optimizer) in enumerate(zip(losses, optimizers)):
            metric_logger.update(**{'loss{}'.format(group): loss.item()})
            metric_logger.update(**{'lr{}'.format(group): optimizer.param_groups[0]["lr"]})
    print('finish training')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate_network(val_loader, model, linear_classifiers, permutes, lock_rank):
    model.eval()
    buffer_dict = defaultdict(list)
    linear_classifiers.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for data, label in metric_logger.log_every(tqdm(val_loader), 389, header):
        # move to gpu
        inp = data.cuda()
        target = label.cuda()

        # forward
        with torch.no_grad():
            output = model(inp)
        
        losses = []
        acc1s = []
        acc5s = []
        for group, linear_classifier in enumerate(linear_classifiers):
            linear_classifier.eval()
            pred = linear_classifier(output)
            loss = nn.CrossEntropyLoss()(pred, target)
            losses.append(loss)

            if linear_classifier.module.num_labels >= 5:
                acc1, acc5 = utils.accuracy(pred, target, topk=(1, 5))
                acc1s.append(acc1)
                acc5s.append(acc5)
            else:
                acc1, = utils.accuracy(pred, target, topk=(1,))
                acc1s.append(acc1)
                buffer_dict["pred_labels{}".format(group)].append(F.softmax(pred,dim=1)[:,1].cpu().detach())
                print(buffer_dict["pred_labels{}".format(group)])
                # print(F.softmax(output,dim=1)[:,1].cpu().detach().numpy().shape)
                # if group%(len(linear_classifiers))==0:
                #     buffer_dict["real_labels"].append(torch.LongTensor(target.cpu().detach().numpy()))

            batch_size = inp.shape[0]
            metric_logger.update(**{'loss{}'.format(group): loss.item()})
            metric_logger.meters['acc{}@1'.format(group)].update(acc1.item(), n=batch_size)
            if linear_classifier.module.num_labels >= 5:
                metric_logger.meters['acc{}@5'.format(group)].update(acc5.item(), n=batch_size)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, buffer_dict


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # flatten
        # x = x.view(x.size(0), -1)
        # print(x.shape)

        # linear layer
        
        return self.linear(x)

class Camelyon(object):
    def __init__(self, root, list_file, transform,database=False):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.has_labels = len(lines[0].split()) == 2
        # if self.has_labels:
        #     self.fns, self.labels = zip(*[l.strip().split() for l in lines])
        #     self.labels = [int(l) for l in self.labels]
        # else:
        #     self.fns = [l.strip() for l in lines]

        self.fns, self.bags = zip(*[(l.strip().split()[0],l.strip().split('/')[1]) for l in lines])
        self.bags_to_idx = {bag:idx for idx,bag in enumerate(sorted(set(self.bags)))}
        self.labels = [int(1) if patch.split('/')[-3].startswith('tumor') else int(0) for patch in self.fns]
        self.bag_idxs = [self.bags_to_idx[bag] for bag in self.bags]
        self.x_coords = [int(patch.split('_')[-3]) for patch in self.fns]
        self.y_coords = [int(patch.split('_')[-2]) for patch in self.fns]

        self.fns = [os.path.join(root, fn) for fn in self.fns]
        
        self.transform = transform
        self.database =database
        # if database:
        #     self.database = redis.Redis(host='localhost', port=6379)


    # def _init_memcached(self):
    #     if not self.initialized:
    #         assert self.mclient_path is not None
    #         self.mc_loader = McLoader(self.mclient_path)
    #         self.initialized = True

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        if self.database:
            key = str(self.fns[idx])
            # print(key, self.database.get(key))
            img = pickle.loads(self.database.get(key))
            img = self.transform(img)
            target = self.labels[idx]
            return img, target
        img = Image.open(self.fns[idx])
        # print(idx, self.fns[idx])
        img = img.convert('RGB')
        img = self.transform(img)
        target = self.labels[idx]
        return img, target

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base/Large.""")
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO, iBOT).
        we set this to 2 for base/large-size models with [CLS] token when doing linear classification.""")
    parser.add_argument('--arch', default='resnet18', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'resnet18', 'resnet50', 'resnet101', 'dalle_encoder'], help='Architecture.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--window_size', default=7, type=int, help='Window size of the model.')
    parser.add_argument('--pretrained_weights', default='/remote-home/share/GraphMIL/backbone_check/debug/ConCL/simclr.pth', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='', type=str,
        help='Please specify path to the ImageNet data.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="/remote-home/kongyijian/GraphMIL/backbone_check/debug/eval/nctcrc_simclr", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=2, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')
    parser.add_argument('--sweep_lr_only', default=True, type=bool, help='Wether or not to only sweep over learning rate')
    parser.add_argument('--database', default=False, type=bool, help='Wether or not to use database')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_linear(args_copy)
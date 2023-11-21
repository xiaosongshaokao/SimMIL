# -*- coding: utf-8 -*-
import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle

# vit_moco3
import torch, torchvision
from torch.utils.data import Dataset
from functools import partial
import moco.builder_infence
import moco.loader
import moco.optimizer
import torchvision.models as torchvision_models
import vits
#ctranspath
from ctran import ctranspath
#clip
import clip
#transpath
from numpy.lib.function_base import append
from torch.autograd import Variable
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from torchvision.models import resnet50
from byol_pytorch.byol_pytorch_get_feature import BYOL
from benchmark_res50 import R50
from torch.utils.data import Dataset
import os
class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # if img.size[0] != 256:
        #     print(temp_path, img.size)
        sample = {'input': img}
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def Norbag_dataset(args, csv_file_path):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = mean, std = std)
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)
def Norbag_dataset_256(args, csv_file_path):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        transforms.Resize(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = mean, std = std)
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)
def MoCobag_dataset(args, csv_file_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    print('MOCO')
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose(augmentation))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)
    
from PIL import ImageFilter
import random
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single'):
    i_classifier.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    bag_len = []
    for i in range(0, num_bags):
        feats_list = []
        # if magnification=='single' or magnification=='low':
        #     csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        # elif magnification=='high':
        #     csv_file_path = glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpg')) + glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpeg'))
        WSI_path = bags_list[i]
        print('\n',WSI_path)
        if 'Camelyon' in  args.dataset:
            split = WSI_path.split('/')[-2].split('_')[0]
            WSI_name = WSI_path.split('/')[-1].split('.')[0]
            if args.mag == 20:
                # csv_file_path =  glob.glob(os.path.join('patches', split, '*', WSI_name,'*.png')) # my preprocess of 20x tcga  exludes HE/NOR patches OMG！
                # csv_file_path =  glob.glob(os.path.join('patches', split, 'neg_256_20X_Nor', WSI_name,'*.png')) + glob.glob(os.path.join('patches', split, 'nontumor_256_20X_Nor', WSI_name,'*.png')) + glob.glob(os.path.join('patches', split, 'tumor_256_20X_Nor', WSI_name,'*.png'))
                
                csv_file_path =  glob.glob(os.path.join('patches', split, 'neg_256_20X', WSI_name,'*.png')) + glob.glob(os.path.join('patches', split, 'nontumor_256_20X', WSI_name,'*.png')) + glob.glob(os.path.join('patches', split, 'tumor_256_20X', WSI_name,'*.png'))
                # csv_file_path =  glob.glob(os.path.join('patches', 'train', 'neg_256_20X_Nor', WSI_name,'*.png')) + glob.glob(os.path.join('patches', 'train', 'nontumor_256_20X_Nor', WSI_name,'*.png')) + glob.glob(os.path.join('patches', 'train', 'tumor_256_20X_Nor', WSI_name,'*.png'))
                # csv_file_path =  glob.glob(os.path.join('patches', 'test', 'neg_256_20X_Nor', WSI_name,'*.png')) + glob.glob(os.path.join('patches', 'test', 'nontumor_256_20X_Nor', WSI_name,'*.png')) + glob.glob(os.path.join('patches', 'test', 'tumor_256_20X_Nor', WSI_name,'*.png')) 
                # csv_file_path =  glob.glob(os.path.join( 'WSI/Camelyon16/single','*',WSI_name,'*.jpeg')) #rubbish default
            elif args.mag == 5:
                csv_file_path =  glob.glob(os.path.join('patches_5x', split, '*', WSI_name,'*.png')) # my preprocess of 5x tcga
        elif 'tcga' in  args.dataset and (magnification=='single' or magnification=='low'):
            csv_file_path = glob.glob(os.path.join('WSI',bags_list[i], '*.jpg')) + glob.glob(os.path.join('WSI',bags_list[i], '*.jpeg'))
        elif 'tcga' in  args.dataset and magnification=='high':
            csv_file_path = glob.glob(os.path.join('WSI',bags_list[i], '*'+os.sep+'*.jpg')) + glob.glob(os.path.join('WSI',bags_list[i], '*'+os.sep+'*.jpeg'))
        elif 'brca' in args.dataset:
            WSI_name = WSI_path.split('.')[0]
            csv_file_path = glob.glob(os.path.join('/remote-home/share/songyicheng/brca_dataset/', WSI_name ,'*.png'))
        elif '10p_C' in args.dataset:
            split = WSI_path.split('/')[-2].split('_')[0]
            WSI_name = WSI_path.split('/')[-1].split('.')[0]
            if args.mag == 20:
                csv_file_path =  glob.glob(os.path.join('patches', split, 'neg_256_20X', WSI_name,'*.png')) + glob.glob(os.path.join('patches', split, 'nontumor_256_20X', WSI_name,'*.png')) + glob.glob(os.path.join('patches', split, 'tumor_256_20X', WSI_name,'*.png'))
        elif '10p_T' in args.dataset:
            csv_file_path = glob.glob(os.path.join('WSI',bags_list[i], '*'+os.sep+'*.jpg')) + glob.glob(os.path.join('WSI',bags_list[i], '*'+os.sep+'*.jpeg'))
        elif '10p_B' in args.dataset:
            WSI_name = WSI_path.split('.')[0]
            csv_file_path = glob.glob(os.path.join('/remote-home/share/songyicheng/brca_dataset/', WSI_name ,'*.png'))
        bag_len.append(len(csv_file_path))

        if args.backbone == 'vit_small' or args.backbone == 'ctranspath':
            dataloader, bag_size = Norbag_dataset(args, csv_file_path)
            with torch.no_grad():
                for iteration, batch in enumerate(dataloader):
                    patches = batch['input'].float().cuda() 
                    feats = i_classifier(patches)
                    feats = feats.cpu().numpy()
                    feats_list.extend(feats)
                    sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        elif args.backbone == 'transpath':
            dataloader, bag_size = Norbag_dataset_256(args, csv_file_path)
            # dataloader, bag_size = Norbag_dataset(args, csv_file_path)
            with torch.no_grad():
                for iteration, batch in enumerate(dataloader):
                    patches = batch['input'].float().cuda() 
                    # feats = i_classifier(patches)
                    _, feats = i_classifier(patches,return_embedding = True)
                    feats = feats.cpu().numpy()
                    feats_list.extend(feats)
                    sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        elif args.backbone == 'clip':
            dataloader, bag_size = Norbag_dataset(args, csv_file_path)
            with torch.no_grad():
                for iteration, batch in enumerate(dataloader):
                    patches = batch['input'].float().cuda() 
                    feats = i_classifier.encode_image(patches)
                    feats = feats.cpu().numpy()
                    feats_list.extend(feats)
                    sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        elif args.backbone == 'resnetTrunk':
            dataloader, bag_size = Norbag_dataset(args, csv_file_path)
            with torch.no_grad():
                for iteration, batch in enumerate(dataloader):
                    patches = batch['input'].float().cuda() 
                    feats = i_classifier(patches)
                    feats = feats.cpu().numpy()
                    feats_list.extend(feats)
                    sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        else:
            dataloader, bag_size = Norbag_dataset(args, csv_file_path)
            # dataloader, bag_size = MoCobag_dataset(args, csv_file_path)
            with torch.no_grad():
                for iteration, batch in enumerate(dataloader):
                    patches = batch['input'].float().cuda() 
                    feats, classes = i_classifier(patches)
                    feats = feats.cpu().numpy()
                    feats_list.extend(feats)
                    sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            
            if 'Camelyon' in  args.dataset or '10p_C' in args.dataset:
                os.makedirs(os.path.join(save_path, split), exist_ok=True)
                df.to_csv(os.path.join(save_path, split, bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
            elif 'tcga' in  args.dataset or '10p_T' in args.dataset:
                os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
                df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
            elif 'brca' in args.dataset or '10p_B' in args.dataset:
                os.makedirs(os.path.join(save_path, 'brca_dataset'), exist_ok=True)
                df.to_csv(os.path.join(save_path, 'brca_dataset', WSI_name + '.csv'), index=False, float_format='%.4f')
            
            print('\n')    

    print('Bag length max:{}, mean{}'.format(max(bag_len),sum(bag_len)/len(bag_len)))
def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None, fusion='fusion'):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(0, num_bags): 
            low_patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
            for idx, low_patch in enumerate(low_patches):
                high_folder = os.path.dirname(low_patch) + os.sep + os.path.splitext(os.path.basename(low_patch))[0]
                high_patches = glob.glob(high_folder+os.sep+'*.jpg') + glob.glob(high_folder+os.sep+'*.jpeg')
                if len(high_patches) == 0:
                    pass
                else:
                    for high_patch in high_patches:
                        img = Image.open(high_patch)
                        img = VF.to_tensor(img).float().cuda()
                        feats, classes = embedder_high(img[None, :])
                        if fusion == 'fusion':
                            feats = feats.cpu().numpy()+0.25*feats_list[idx]
                        if fusion == 'cat':
                            feats = np.concatenate((feats.cpu().numpy(), feats_list[idx][None, :]), axis=-1)
                        feats_tree_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, idx+1, len(low_patches)))
            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + bags_list[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
                df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
            print('\n')            


def del_patch(csv_file_path):
    for idx, img_path in enumerate(csv_file_path):
        img = Image.open(img_path)
        if img.size[0] != 256:
            print('Removing',img_path)
            csv_file_path.pop(idx)
            os.remove(img_path)

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=32, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='batch', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='high', type=str, help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights', default=None, type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--weights_high', default=None, type=str, help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default=None, type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    parser.add_argument('--dataset', default='TCGA-lung-single', type=str, help='Dataset folder name [TCGA-lung-single]')
    parser.add_argument('--mag', default=20, type=int, help='magnitude')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small',
                    help='model architecture: '
                         ' (default: vit_small)')
    parser.add_argument('--weight_path', default='TCGA-lung-single', type=str, help='weight path')
    args = parser.parse_args()
    # gpu_ids = tuple(args.gpu_index) pretrained_weights/imagenet_r18.pth
    # os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    if args.norm_layer == 'instance':
        norm=nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':  
        norm=nn.BatchNorm2d
        if args.weights == 'ImageNet':
            pretrain = True
        else:
            pretrain = False

    if args.backbone == 'vit_small':
        model = moco.builder_infence.MoCo_ViT(partial(vits.__dict__[args.arch], stop_grad_conv1=True))
        pretext_model = torch.load(r'pretrained_weights/vit_small.pth.tar')['state_dict']
        print('Use pretrained features from mocov3 vit ')
        model.base_encoder.head = nn.Identity()
        model.predictor= nn.Identity()
        model = nn.DataParallel(model).cuda()
        msg = model.load_state_dict(pretext_model, strict=False)
        print(msg.missing_keys)
        model.eval()
        num_feats = 384
    elif args.backbone == 'ctranspath':
        model = ctranspath()
        model.head = nn.Identity()
        num_feats = 768
    elif args.backbone == 'clip':
        model, preprocess = clip.load("RN50") # MoCoV2 SwAV
        num_feats = 1024
    elif args.backbone == 'resnetTrunk':
        model = R50(pretrained=True, progress=False, key='MoCoV2')
        num_feats = 1024
    elif args.backbone == 'transpath':
        model = BYOL(
            image_size=256,
            hidden_layer='to_latent'
        )
        pretext_model = torch.load(r'pretrained_weights/checkpoint_transpath.pth')
        model = nn.DataParallel(model).cuda()
        model.load_state_dict(pretext_model, strict=True)

        model.module.online_encoder.net.head = nn.Identity()

        model.eval()
    # else:
    resnet = 0
    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    # if args.backbone == 'sslp3':
    #     resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
    #     num_feats = 512
    elif args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    # if args.backbone == 'resnet101':
    #     resnet = models.resnet101(pretrained=pretrain, norm_layer=norm)
    #     num_feats = 2048
    if resnet:
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()
        for k in resnet.state_dict():
            print(k)
    
    
    if args.magnification == 'tree' and args.weights_high != None and args.weights_low != None:
        i_classifier_h = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
        i_classifier_l = mil.IClassifier(copy.deepcopy(resnet), num_feats, output_class=args.num_classes).cuda()
        
        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights== 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                raise ValueError('Please use batch normalization for ImageNet feature')
        else:
            weight_path = os.path.join('simclr', 'runs', args.weights_high, 'checkpoints', 'model.pth')
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_h.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_h.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder-high.pth'))

            weight_path = os.path.join('simclr', 'runs', args.weights_low, 'checkpoints', 'model.pth')
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_l.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_l.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder-low.pth'))
            print('Use pretrained features.')


    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':  
        if args.backbone == 'vit_small' or args.backbone == 'ctranspath' or args.backbone == 'transpath':
            weight_path = args.weight_path
            td = torch.load(weight_path)
            
            msg = model.load_state_dict(td['state_dict'], strict=True)
            i_classifier = nn.DataParallel(model).cuda()
            print(msg)
            print('Use pretrained features from: ', weight_path)
        elif args.backbone == 'clip' or args.backbone == 'resnetTrunk':
            i_classifier = model.cuda()
            print('Use pretraind features from clip or resnetTrunk')
        else:
            i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()

            if args.weights == 'ImageNet':
                if args.norm_layer == 'batch':
                    print('Use ImageNet features.')
                else:
                    print('Please use batch normalization for ImageNet feature')
            else:
                # if args.weights is not None:
                #     weight_path = os.path.join('simclr', 'runs', args.weights, 'checkpoints', 'model.pth')
                # else:
                #     weight_path = glob.glob('simclr/runs/*/checkpoints/*.pth')[-1]

                weight_path = args.weight_path #default c16 emb
                # # pretrained_weights/model-v0-tcga.pth
                state_dict_weights = torch.load(weight_path)['state_dict']
                # state_dict_weights = torch.load(weight_path)['backbone']
                # for i in range(4):
                #     state_dict_weights.popitem()
                state_dict_init = i_classifier.state_dict()
                new_state_dict = OrderedDict()
                for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                    name = k_0
                    new_state_dict[name] = v
                msg = i_classifier.load_state_dict(new_state_dict, strict=False)
                i_classifier = nn.DataParallel(i_classifier).cuda()
                print(msg)
                # os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
                # torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder.pth'))
                print('Use pretrained features from: ', weight_path)    
    
    # if args.magnification == 'tree' or args.magnification == 'low' or args.magnification == 'high' :
    #     bags_path = os.path.join('WSI', args.dataset, 'pyramid', '*', '*')
    # else:
        # bags_path = os.path.join('WSI', args.dataset, 'single', '*', '*')
    if 'Camelyon' in  args.dataset:
        bags_path = os.path.join('WSI','Camelyon16', '*', '*')
        bags_list = glob.glob(bags_path)
    elif 'brca' in args.dataset:
        bags_path = pd.read_csv('/remote-home/share/songyicheng/brca_dataset/BRCA.csv')
        bags_list = bags_path.iloc[:,0].tolist()
    elif '10p_C' in args.dataset:
        bags_path = os.path.join('WSI','Camelyon16', '*', '*')
        bags_list_full = glob.glob(bags_path)
        bags_list = []
        with open('/remote-home/share/songyicheng/10p_datasets/10p_C16.txt') as f:
            img_list = f.readlines()
            for bag in bags_list_full:
                name = bag.split('/')[-1].split('.')[0]
                for img in img_list:
                    if name in img:
                        bags_list.append(bag)
                    else:
                        continue
    elif '10p_T' in args.dataset:
        bags_path = pd.read_csv('datasets/tcga-dataset/TCGA.csv')
        bags_list_full = bags_path.iloc[:,0].tolist()
        bags_list = []
        with open('/remote-home/share/songyicheng/10p_datasets/10p_TCGA.txt') as f:
            img_list = f.readlines()
            for bag in bags_list_full:
                name = bag.split('/')[-1]
                for img in img_list:
                    if name in img:
                        bags_list.append(bag)
                    else:
                        continue
    elif '10p_B' in args.dataset:
        bags_path = pd.read_csv('/remote-home/share/songyicheng/brca_dataset/BRCA.csv')
        bags_list_full = bags_path.iloc[:,0].tolist()
        bags_list = []
        with open('/remote-home/share/songyicheng/10p_datasets/10p_BRCA.txt') as f:
            img_list = f.readlines()
            for bag in bags_list_full:
                name = bag.split('.')[0]
                for img in img_list:
                    if name in img:
                        bags_list.append(bag)
                    else:
                        continue
    else:
        bags_path = pd.read_csv('datasets/tcga-dataset/TCGA.csv')
        bags_list = bags_path.iloc[:,0].tolist()

    feats_path = os.path.join('datasets', args.dataset)
        
    os.makedirs(feats_path, exist_ok=True)
    
    
    if args.magnification == 'tree':
        compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path, 'cat')
    else:
        compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)
    n_classes = glob.glob(os.path.join('datasets', args.dataset, '*'+os.path.sep))
    n_classes = sorted(n_classes)
    all_df = []
    if 'Camelyon' in  args.dataset:
        bags_csv='datasets/Camelyon16/Camelyon16.csv'
        df = pd.read_csv(bags_csv)
        ref_label = {}
        for idx in range(len(df)):
            name = df.loc[idx][0].split('/')[-1].split('.')[0] +'.csv'
            label= df.loc[idx][1]
            ref_label[name] = label
        for i, item in enumerate(n_classes):
            bag_csvs = glob.glob(os.path.join(item, '*.csv'))
            bag_df = pd.DataFrame(bag_csvs)
            bag_df['label'] = i
            for idx in range(len(bag_df)):
                name = bag_df.loc[idx][0].split('/')[-1].split('.')[0] +'.csv'
                bag_df.iloc[idx,1] = ref_label[name]
            bag_df.to_csv(os.path.join('datasets', args.dataset, item.split(os.path.sep)[2]+'.csv'), index=False)
            all_df.append(bag_df)
    elif '10p_C' in args.dataset:
        bags_csv='datasets/Camelyon16/Camelyon16.csv'
        df = pd.read_csv(bags_csv)
        ref_label = {}
        for idx in range(len(df)):
            name = df.loc[idx][0].split('/')[-1].split('.')[0] +'.csv'
            label= df.loc[idx][1]
            ref_label[name] = label
        for i, item in enumerate(n_classes):
            bag_csvs = glob.glob(os.path.join(item, '*.csv'))
            bag_df = pd.DataFrame(bag_csvs)
            bag_df['label'] = i
            for idx in range(len(bag_df)):
                name = bag_df.loc[idx][0].split('/')[-1].split('.')[0] +'.csv'
                bag_df.iloc[idx,1] = ref_label[name]
            bag_df.to_csv(os.path.join('datasets', args.dataset, item.split(os.path.sep)[2]+'.csv'), index=False)
            all_df.append(bag_df)
    elif 'tcga' in args.dataset:
        bags_csv='datasets/tcga-dataset/TCGA.csv'
        df = pd.read_csv(bags_csv)
        for idx in range(len(df)):
            df.iloc[idx,0] =  os.path.join(feats_path, df.iloc[idx,0])
        
        all_df.append(df)
    elif '10p_T' in args.dataset:
        bags_csv='/remote-home/share/songyicheng/10p_datasets/10p_TCGA.csv'
        df = pd.read_csv(bags_csv)
        for idx in range(len(df)):
            df.iloc[idx,0] =  os.path.join(feats_path, df.iloc[idx,0])
        
        all_df.append(df)
    elif 'brca' in args.dataset:
        bag_csvs = '/remote-home/share/songyicheng/brca_dataset/BRCA.csv'
        df = pd.read_csv(bag_csvs)
        for idx in range(len(df)):
            df.iloc[idx, 0] = os.path.join(feats_path,'brca_dataset', df.iloc[idx, 0].split('.')[0] + '.csv')
        
        all_df.append(df)
    elif '10p_B' in args.dataset:
        bag_csvs = '/remote-home/share/songyicheng/10p_datasets/10p_BRCA.csv'
        df = pd.read_csv(bag_csvs)
        for idx in range(len(df)):
            df.iloc[idx, 0] = os.path.join(feats_path,'brca_dataset', df.iloc[idx, 0] + '.csv')
        
        all_df.append(df)

    
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    # bags_path = shuffle(bags_path)
    bags_path.to_csv(os.path.join('datasets', args.dataset, args.dataset+'.csv'), index=False)
    
if __name__ == '__main__':
    main()
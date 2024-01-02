# -*- coding: utf-8 -*-
import enum
import re
from symbol import testlist_star_expr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset 
import redis
import pickle
import time 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve
import random 
import torch.backends.cudnn as cudnn
import json
torch.multiprocessing.set_sharing_strategy('file_system')
import os

os.environ['TMPDIR'] = '/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/tmp'
class Unitopatho(Dataset):
    def __init__(self,fold,database=None):
        super(Unitopatho).__init__()
        self.fold = fold
        self.database = database
        self.fold_path = '/remote-home/share/promptMIL/datasets/Unitopatho/{}/{}_npy'.format(self.fold, self.fold)
        self.feat_list = os.listdir(self.fold_path)
        
    def __getitem__(self,idx):
        feat_path = os.path.join(self.fold_path, self.feat_list[idx])
        if self.database is not None:
            feats = pickle.loads(self.database.get(feat_path+'feats'))
            label = pickle.loads(self.database.get(feat_path+'label'))
            return  label ,feats
        print(feat_path)
        label = self.feat_list[idx].split('_')[1].split('.')[0]
        label = torch.LongTensor([float(label)])
        feat = np.load(feat_path)
        feat_tensor = torch.Tensor(feat)
        return feat_tensor, label
    
    def __len__(self):
        return len(self.feat_list)


class BagDataset(Dataset):
    def __init__(self,train_path, args, database=None, return_slide_index=False):
        super(BagDataset).__init__()
        self.train_path = train_path
        self.args = args
        self.database = database
        self.return_slide_index = return_slide_index

    def get_bag_feats(self,csv_file_df, args):
        if args.dataset.startswith('tcga'):
            feats_csv_path = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/datasets',args.dataset,'data_tcga_lung_tree' ,csv_file_df.iloc[0].split('/')[-1] + '.pt')
        elif args.dataset.startswith('4ktcga'):
            feats_csv_path = os.path.join('/remote-home/share/songyicheng/HIPT_datasets',args.dataset,'tcga_4096' ,csv_file_df.iloc[0].split('/')[-1] + '.pt')
        elif args.dataset.startswith('4kbrca'):
            feats_csv_path = os.path.join('/remote-home/share/songyicheng/HIPT_datasets',args.dataset,'brca_4096' ,csv_file_df.iloc[0].split('/')[-1] + '.pt')
        else:
            feats_csv_path = csv_file_df.iloc[0]
        if self.database is None:
            # 从.pt文件中读取数据
            feats_tensor = torch.load(feats_csv_path)

            # 将张量转换为Pandas DataFrame
            df = pd.DataFrame(feats_tensor.numpy())
            # df = pd.read_csv(feats_csv_path)
            feats = shuffle(df).reset_index(drop=True)
            feats = feats.to_numpy()
            label = np.zeros(args.num_classes)
            if args.num_classes==1:
                label[0] = csv_file_df.iloc[1]
            else:
                if int(csv_file_df.iloc[1])<=(len(label)-1):
                    label[int(csv_file_df.iloc[1])] = 1
            label = torch.tensor(np.array(label))
            feats = torch.tensor(np.array(feats)).float()
        else:
            key = csv_file_df.iloc[0]
            feats = pickle.loads(self.database.get(key+'feats'))
            label = pickle.loads(self.database.get(key+'label'))

        return label, feats

    def dropout_patches(self,feats, p):
        idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
        sampled_feats = np.take(feats, idx, axis=0)
        pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
        pad_feats = np.take(sampled_feats, pad_idx, axis=0)
        sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
        return sampled_feats
    
    def __getitem__(self, idx):
        label, feats = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        if self.return_slide_index:
            return  label, feats, idx
        else:
            return  label, feats
        
    def __len__(self):
        return len(self.train_path)


class BagDataset_online(Dataset):
    def __init__(self,train_path, args, database=None):
        super(BagDataset_online).__init__()
        self.train_path = train_path
        # self.csv_file_df = csv_file_df
        self.args = args
        self.database = database
        import torchvision.models as models
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.encoder = self.encoder.cuda()


    def get_bag_feats(self,csv_file_df, args):
        if args.dataset == 'TCGA-lung-default':
            feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
        elif args.dataset.startswith('tcga'):
            feats_csv_path = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/datasets',args.dataset,'data_tcga_lung_tree' ,csv_file_df.iloc[0].split('/')[-1] + '.csv')
        else:
            feats_csv_path = csv_file_df.iloc[0]
        split = feats_csv_path.split('/')[-2].split('_')[0]
        WSI_name = feats_csv_path.split('/')[-1].split('.')[0]
        csv_file_path =  glob.glob(os.path.join('patches', split, '*', WSI_name,'*.png'))
        dataloader, bag_size = MoCobag_dataset(args, csv_file_path)
        self.encoder.eval()
        feats_list = []
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                feats = self.encoder(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
        label = np.zeros(args.num_classes)
        if args.num_classes==1:
            label[0] = csv_file_df.iloc[1]
        else:
            if int(csv_file_df.iloc[1])<=(len(label)-1):
                label[int(csv_file_df.iloc[1])] = 1
        label = torch.tensor(np.array(label))
        feats = torch.tensor(np.array(feats_list)).float()
        return label, feats

    def dropout_patches(self,feats, p):
        idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
        sampled_feats = np.take(feats, idx, axis=0)
        pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
        pad_feats = np.take(sampled_feats, pad_idx, axis=0)
        sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
        return sampled_feats
    
    def __getitem__(self, idx):
        label, feats = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        return  label, feats
        
    def __len__(self):
        return len(self.train_path)
def MoCobag_dataset(args, csv_file_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
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
    transformed_dataset = SimpleBagDataset(csv_file=csv_file_path,
                                    transform=Compose(augmentation))
    dataloader = DataLoader(transformed_dataset, batch_size=256, shuffle=False, num_workers=8, drop_last=False)
    return dataloader, len(transformed_dataset)
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
from PIL import Image
class SimpleBagDataset():
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
        sample = {'input': img}
        
        
        return sample 
from PIL import ImageFilter
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
def train(train_df, milnet, criterion, optimizer, args, log_path, epoch=0):
    milnet.train()
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0
    
    for i,(bag_label,bag_feats) in enumerate(train_df):
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)  # n x feat_dim
        optimizer.zero_grad()
        if args.model == 'dsmil':
            ins_prediction, bag_prediction, attention, atten_B= milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)      
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
        elif args.model in ['abmil', 'max', 'mean']:
            bag_prediction, _, attention = milnet(bag_feats)
            loss =  criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        elif args.model in ['max_pooling', 'mean_pooling']:
            bag_prediction = milnet(bag_feats)
            loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        elif args.model == 'clam':
            if args.num_classes == 2:
                logits, bag_prediction, Y_hat, A_raw, results_dict, attention = milnet(bag_feats, bag_label, instance_eval=True)
            elif args.num_classes == 1:
                logits, bag_prediction, Y_hat, A_raw, results_dict, attention = milnet(bag_feats, bag_label, instance_eval=False)
            if 'instance_loss' in results_dict.keys():
                instance_loss = results_dict['instance_loss']
            else:
                instance_loss = None
            if args.num_classes == 2:
                bag_loss = criterion(logits.view(1, -1), bag_label.view(1, -1).argmax(dim=-1))
            elif args.num_classes == 1:
                bag_loss = criterion(logits.view(1, -1), bag_label.view(1, -1))
            if instance_loss:
                loss = 0.7*bag_loss+0.3*instance_loss
            else:
                loss = bag_loss
        elif args.model == 'hipt':
            logits, bag_prediction, Y_hat, attention = milnet(bag_feats)
            loss = criterion(logits.view(1, -1), bag_label.view(1, -1).argmax(dim=-1)) / 32
       
        loss.backward()
        # cancel gradients for the confounders 
        if args.c_learn:
            if epoch<args.freeze_epoch:
                for name, p in milnet.named_parameters():
                    if "confounder_feat" in name:
                        p.grad = None
        optimizer.step()
        total_loss = total_loss + loss.item()
        if not args.model in ['max_pooling', 'mean_pooling']:
            atten_max = atten_max + attention.max().item()
            atten_min = atten_min + attention.min().item()
            atten_mean = atten_mean +  attention.mean().item()
            
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f ,attention max:%.4f, min:%.4f, mean:%.4f' % (i, len(train_df), loss.item(), 
                            attention.max().item(), attention.min().item(), attention.mean().item()))
        else:
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    if not args.model in ['max_pooling', 'mean_pooling']:
        atten_max = atten_max / len(train_df)
        atten_min = atten_min / len(train_df)
        atten_mean = atten_mean / len(train_df)
        if not args.prompt_type and not args.lp:
            with open(log_path,'a+') as log_txt:
                    log_txt.write('\n atten_max'+str(atten_max))
                    log_txt.write('\n atten_min'+str(atten_min))
                    log_txt.write('\n atten_mean'+str(atten_mean))
    return total_loss / len(train_df)


def test(test_df, milnet, criterion, optimizer, args, log_path, epoch):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i,(bag_label,bag_feats) in enumerate(test_df):
            label = bag_label.numpy()
            bag_label = bag_label.cuda()
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)
            if args.model == 'dsmil':
                ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
                max_prediction, _ = torch.max(ins_prediction, 0)
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
                loss = 0.5*bag_loss + 0.5*max_loss
            elif args.model in ['abmil', 'max', 'mean']:
                bag_prediction, _, _ =  milnet(bag_feats)
                max_prediction = bag_prediction
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            elif args.model in ['max_pooling', 'mean_pooling']:
                bag_prediction = milnet(bag_feats)
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_prediction = bag_prediction
            elif args.model == 'clam':
                if args.num_classes == 2:
                    logits, bag_prediction, Y_hat, A_raw, results_dict, attention = milnet(bag_feats, bag_label, instance_eval=True)
                elif args.num_classes == 1:
                    logits, bag_prediction, Y_hat, A_raw, results_dict, attention = milnet(bag_feats, bag_label, instance_eval=False)
                if 'instance_loss' in results_dict.keys():
                    instance_loss = results_dict['instance_loss']
                else:
                    instance_loss = None
                if args.num_classes == 2:
                    bag_loss = criterion(logits.view(1, -1), bag_label.view(1, -1).argmax(dim=-1))
                elif args.num_classes == 1:
                    bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_prediction = bag_prediction
                if instance_loss:
                    loss = 0.7*bag_loss+0.3*instance_loss
                else:
                    loss = bag_loss
            elif args.model == 'hipt':
                logits, bag_prediction, Y_hat, attention = milnet(bag_feats)
                loss = criterion(logits.view(1, -1), bag_label.view(1, -1).argmax(dim=-1)) / 32
                max_prediction = bag_prediction
           
           
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend(label)
            if args.average:   # notice args.average here
                test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
                
            else: test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    # fixed threshold = 0.5
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    test_predictions_= test_predictions>0.5
    acc = accuracy_score(test_labels, test_predictions_)
    cls_report = classification_report(test_labels, test_predictions_, digits=4)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n *****************Threshold by 0.5*****************')
    if args.num_classes==1:
        print('\n', confusion_matrix(test_labels,test_predictions_))
        info = confusion_matrix(test_labels,test_predictions_) 
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
    else:
        for i in range(args.num_classes):
            print('\n', confusion_matrix(test_labels[:,i],test_predictions_[:,i]))
            info = confusion_matrix(test_labels[:,i],test_predictions_[:,i]) 
            with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
    print('Accuracy', acc)
    print('\n', cls_report)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n Accuracy:'+str(acc))
        log_txt.write('\n'+cls_report)

    # chosing threshold
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print(confusion_matrix(test_labels,test_predictions))
        info = confusion_matrix(test_labels,test_predictions)
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
        
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
            print(confusion_matrix(test_labels[:,i],test_predictions[:,i]))
            info = confusion_matrix(test_labels[:,i],test_predictions[:,i])
            with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
            #print(class_prediction_bag.shape)
            # cls_report = classification_report(test_labels[:,i], class_prediction_bag, output_dict=True, zero_division=0)
            # print(cls_report)
    bag_score = 0
    # average acc of all labels
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)  #ACC
    cls_report = classification_report(test_labels, test_predictions, digits=4)

    # print(confusion_matrix(test_labels,test_predictions))
    print('\n dsmil-metrics: multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
    print('\n', cls_report)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n dsmil-metrics: multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
        log_txt.write('\n' + cls_report)
    if epoch == args.num_epochs-1:
        log_rep = classification_report(test_labels, test_predictions, digits=4,output_dict=True)
        with open(log_path,'a+') as log_txt:
            log_txt.write('{:.2f},{:.2f},{:.2f},{:.2f} \n'.format(log_rep['macro avg']['precision']*100,log_rep['macro avg']['recall']*100,avg_score*100,sum(auc_value)/len(auc_value)*100))
    # y_pred, y_true = inverse_convert_label(test_predictions), inverse_convert_label(test_labels)
    # p = precision_score(y_true, y_pred, average='macro')
    # r = recall_score(y_true, y_pred, average='macro')
    # acc = accuracy_score(y_true, y_pred)
    # print('\n remix 10-Inverted: Pre:{:.2f},Rec:{:.2f}, Accuracy:{:.2f}'.format(p*100, r*100, acc*100))
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        if sum(label)==0:
            continue
        prediction = predictions[:, c]
        # print(label, prediction,label.shape, prediction.shape, labels.shape, predictions.shape)
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--prompt_type', type=str,help='Using prompt')
    parser.add_argument('--database', action='store_true', help='Using database')
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--lp', action='store_true', help='Linear probing only')
    parser.add_argument('--online', action='store_true', help='Using online patch feeding')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    # parser.add_argument('--dir', type=str,help='directory to save logs')
    parser.add_argument('--agg', type=str,help='which agg')
    parser.add_argument('--dir', type=str,help='directory to save logs')
    parser.add_argument('--n_shot', type=str,help='few shot learning if not none')
    parser.add_argument('--fs_seed', type=str,help='seed for few shot learning if not none')
    parser.add_argument('--static', type=int, nargs='+', default=(0,), help='max:0, mean:1,var:2,min:3')
    parser.add_argument('--c_path', nargs='+', default=None, type=str,help='directory to confounders')
    parser.add_argument('--c_learn', action='store_true', help='learn confounder or not')
    parser.add_argument('--c_dim', default=128, type=int, help='Dimension of the projected confounders')
    parser.add_argument('--freeze_epoch', default=999, type=int, help='freeze confounders during this many epoch from the start')
    parser.add_argument('--c_merge', type=str, default='cat', help='cat or add or sub')
    parser.add_argument('--shuffle_idx', default=-1, type=int, help='[-1|0|1|2]')

    args = parser.parse_args()

    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    if args.lp:
        save_path = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/weights', datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_lp')
    elif args.prompt_type:
        save_path = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/weights', datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_'+str(args.prompt_type))
    elif args.c_path:
        save_path = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/weights', datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_c_path')
    else:
        save_path = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/weights', datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_fulltune')
    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file,'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'
    

    '''
    model 
    1. set require_grad    
    2. choose model and set the trainable params 
    3. load init
    '''
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    if args.lp:
        print('***********Linear probing******************')
        if args.model == 'dsmil':
            import dsmil as mil
            i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
            b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
            milnet = mil.MILNet(i_classifier, b_classifier).cuda()
                # freeze all layers but the last fc
            for name, param in milnet.named_parameters():
                if name not in ['i_classifier.fc.0.weight', 'i_classifier.fc.0.bias', 'b_classifier.fcc.weight', 'b_classifier.fcc.bias']:
                    param.requires_grad = False
                    print('not training {}'.format(name))
                    with open(log_path,'a+') as log_txt:
                        log_txt.write('\n Not training {}'.format(name))
                else:
                    print('Training {}'.format(name))
                    with open(log_path,'a+') as log_txt:
                        log_txt.write('\n Training {}'.format(name))
        elif args.model == 'abmil':
            import abmil as mil
            milnet = mil.Attention(in_size=args.feats_size, out_size=args.num_classes, confounder_learn=args.c_learn).cuda()
            for name, param in milnet.named_parameters():
                if name not in ['classifier.weight', 'classifier.bias']:
                    param.requires_grad = False
                    print('not training {}'.format(name))
                    with open(log_path,'a+') as log_txt:
                        log_txt.write('\n Not training {}'.format(name))
                else:
                    print('Training {}'.format(name))
                    with open(log_path,'a+') as log_txt:
                        log_txt.write('\n Training {}'.format(name))


    else:
        print('*********Full tuning**********')
        if args.model == 'dsmil':
            import dsmil as mil
            i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
            b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity,confounder_path=args.c_path).cuda()
            milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        elif args.model == 'abmil':
            import abmil as mil
            milnet = mil.Attention(in_size=args.feats_size, out_size=args.num_classes,confounder_path=args.c_path, \
                confounder_learn=args.c_learn, confounder_dim=args.c_dim, confounder_merge=args.c_merge).cuda()
        elif args.model == 'max_pooling':
            from max_pooling import max_pooling
            milnet = max_pooling(input_size=args.feats_size).cuda()
        elif args.model == 'mean_pooling':
            from mean_pooling import mean_pooling
            milnet = mean_pooling(input_size=args.feats_size).cuda()
        elif args.model == 'clam':
            from clam import CLAM_SB as CLAM_SB
            assert args.num_classes == 2
            milnet = CLAM_SB(in_size=args.feats_size, n_classes=args.num_classes).cuda()
            milnet.relocate()
        elif args.model == 'hipt':
            from hipt import HIPT_GP_FC as HIPT_GP_FC
            assert args.num_classes == 2
            milnet = HIPT_GP_FC(path_input_dim = args.feats_size, n_classes = args.num_classes).cuda()
            milnet.relocate()

        for name, _ in milnet.named_parameters():
                print('Training {}'.format(name))
                with open(log_path,'a+') as log_txt:
                    log_txt.write('\n Training {}'.format(name))


   
    
    #dataset
    if args.database:
        database = redis.Redis(host='localhost', port=6379)
        print('************************using database************************************')
    else:
        database = None


    if args.dataset.startswith("tcga_subset"):
        bags_csv = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/datasets', args.dataset, args.dataset+'.csv')
        bags_path = pd.read_csv(bags_csv)
        if args.shuffle_idx == -1:
            shuffle_idx = None
        if shuffle_idx:
            bags_path = bags_path.iloc[shuffle_idx, :]
        train_path = bags_path.iloc[0:200, :]
        test_path = bags_path.iloc[200:, :]
        loss_weight = torch.tensor([[ 1, 1]]).cuda()
   
    elif args.dataset.startswith("tcga"):
        bags_csv = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/datasets', args.dataset, args.dataset+'.csv')
        bags_path = pd.read_csv(bags_csv)
        if args.shuffle_idx == -1:
            shuffle_idx = None
        if shuffle_idx:
            bags_path = bags_path.iloc[shuffle_idx, :]
        train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
        test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]
        loss_weight = torch.tensor([[ 1, 1]]).cuda()
    elif args.dataset.startswith('brca'):
        bags_csv = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/datasets', args.dataset, args.dataset+'.csv')
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:656, :]
        test_path = bags_path.iloc[656:, :]
        loss_weight = torch.tensor([[ 1, 1]]).cuda()
    elif args.dataset.startswith('Camelyon16'):
        # bags_csv = os.path.join('datasets', args.dataset, args.dataset+'_off.csv') #offical train test
        bags_csv = os.path.join('/remote-home/share/songyicheng/Code/SimMIL/aggregation/downstream/datasets', args.dataset, args.dataset+'.csv')
        bags_path = pd.read_csv(bags_csv)
        if args.shuffle_idx == -1:
            shuffle_idx = None
        if shuffle_idx is not None:
            bags_path = bags_path.iloc[shuffle_idx, :]
        train_path = bags_path.iloc[129:, :]
        test_path = bags_path.iloc[0:129, :]
        loss_weight = torch.tensor([[1, 1]]).cuda()
    elif args.dataset.startswith('4ktcga'):
        bags_csv = os.path.join('/remote-home/share/songyicheng/HIPT_datasets', args.dataset, args.dataset + '.csv')
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0 : 717, : ]
        test_path = bags_path.iloc[717 : , : ]
        loss_weight = torch.tensor([[ 1, 1]]).cuda()
    elif args.dataset.startswith('4kbrca'):
        bags_csv = os.path.join('/remote-home/share/songyicheng/HIPT_datasets', args.dataset, args.dataset + '.csv')
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0 : 657 , : ]
        test_path = bags_path.iloc[657 : , : ]
        loss_weight = torch.tensor([[ 1, 1]]).cuda()
    if args.online:
        trainset =  BagDataset_online(train_path, args,database)
        train_loader = DataLoader(trainset,1, shuffle=True, num_workers=0)
        testset =  BagDataset(test_path, args,database)
        test_loader = DataLoader(testset,1, shuffle=False, num_workers=16)
    else:
        trainset =  BagDataset(train_path, args,database)
        train_loader = DataLoader(trainset,1, shuffle=True, num_workers=16)
        testset =  BagDataset(test_path, args,database)
        test_loader = DataLoader(testset,1, shuffle=False, num_workers=16)

    # sanity check begins here
    print('*******sanity check *********')
    for k,v in milnet.named_parameters():
        if v.requires_grad == True:
            print(k)

     # loss, optim, schduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight) #pos weight is used to improve the performance of fixed threshold
    if args.model == 'clam':
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'hipt':
        criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    # all_params = milnet.parameters()
    original_params = []
    confounder_parms = []
    for pname, p in milnet.named_parameters():
        if ('confounder' in pname):
            confounder_parms += [p]
            print('confounders:',pname )
        else:
            original_params += [p]
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()), 
    #                             lr=args.lr, betas=(0.5, 0.9), 
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam([
                                {'params':original_params},
                                {'params':confounder_parms, ' weight_decay':args.weight_decay_conf},
                                ], 
                                lr=args.lr, betas=(0.5, 0.9), 
                                weight_decay=args.weight_decay)
    if args.model == 'clam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()), lr=0.0001, weight_decay=1e-5)
    elif args.model == 'hipt':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()), lr=2e-4, weight_decay=1e-5)
  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)


    best_score = 0

    ### test only
    if args.test:
        epoch = args.num_epochs-1
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)   
        
        
        train_loss_bag = 0
        if args.dataset=='TCGA-lung':
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        else:
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        
        if args.model == 'dsmil':
            if  args.agg  == 'tcga':
                load_path = 'test/weights/aggregator.pth' 
            elif  args.agg  == 'c16':
                load_path = 'test-c16/weights/aggregator.pth'   
            else:
                raise NotImplementedError
                
        elif args.model == 'abmil':
            if args.agg  == 'tcga':
                load_path = 'pretrained_weights/abmil_tcgapretrained.pth' # load c-16 pretrain for adaption
            elif args.agg  == 'c16':
                load_path = 'pretrained_weights/abmil_c16pretrained.pth'   # load tcga pretrain for adaption
            else:
                raise NotImplementedError
        state_dict_weights = torch.load(load_path)
        print('Loading model:{} with {}'.format(args.model, load_path))
        with open(log_path,'a+') as log_txt:
            log_txt.write('\n loading init from:'+str(load_path))
        msg = milnet.load_state_dict(state_dict_weights, strict=False)
        print('Missing these:', msg.missing_keys)
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)
        if args.dataset=='TCGA-lung':
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        else:
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        sys.exit()
        
    
    for epoch in range(1, args.num_epochs):
        start_time = time.time()
        train_loss_bag = train(train_loader, milnet, criterion, optimizer, args, log_path, epoch=epoch-1) # iterate all bags
        print('epoch time:{}'.format(time.time()- start_time))
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)
        if args.dataset=='TCGA-lung':
            info = '\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]) +'\n'
            with open(log_path,'a+') as log_txt:
                log_txt.write(info)
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        else:
            info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: '%(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))+'\n'
            with open(log_path,'a+') as log_txt:
                log_txt.write(info)
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        scheduler.step()
        current_score = (sum(aucs) + avg_score)/2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'.pth')
            torch.save(milnet.state_dict(), save_name)
            if args.dataset=='TCGA-lung':
                print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
            else:
                with open(log_path,'a+') as log_txt:
                    info = 'Best model saved at: ' + save_name +'\n'
                    log_txt.write(info)
                    info = 'Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal))+'\n'
                    log_txt.write(info)
                print('Best model saved at: ' + save_name)
                print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
        if epoch == args.num_epochs-1:
            save_name = os.path.join(save_path, 'last.pth')
            torch.save(milnet.state_dict(), save_name)
    log_txt.close()

if __name__ == '__main__':
    main()
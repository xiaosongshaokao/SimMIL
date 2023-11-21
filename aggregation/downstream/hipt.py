import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from os.path import join
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class HIPT_GP_FC(nn.Module):
    def __init__(self, path_input_dim=192, size_arg = "small", dropout=0.25, n_classes=2,
        pretrain_WSI='None'):
        super(HIPT_GP_FC, self).__init__()
        self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
        size = self.size_dict_path[size_arg]

        ### Global Aggregation
        self.pretrain_WSI = pretrain_WSI
        if pretrain_WSI != 'None':
            pass
        else:
            self.global_phi = nn.Sequential(nn.Linear(path_input_dim, 192), nn.ReLU(), nn.Dropout(0.25))
            self.global_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=path_input_dim, nhead=3, dim_feedforward=192, dropout=0.25, activation='relu'
                ), 
                num_layers=2
            )
            self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
            self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])

        self.classifier = nn.Linear(size[1], n_classes)
        

    def forward(self, h_4096, **kwargs):        
        ### Global
        if self.pretrain_WSI != 'None':
            h_WSI = self.global_vit(h_4096.unsqueeze(dim=0))
        else:
            h_4096 = self.global_phi(h_4096)
            h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
            A_4096, h_4096 = self.global_attn_pool(h_4096)  
            A_4096 = torch.transpose(A_4096, 1, 0)
            A_4096 = F.softmax(A_4096, dim=1) 
            h_path = torch.mm(A_4096, h_4096)
            h_WSI = self.global_rho(h_path)

        logits = self.classifier(h_WSI)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return logits, F.softmax(logits, dim=1), Y_hat, A_4096


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            if self.pretrain_WSI != 'None':
                self.global_vit = nn.DataParallel(self.global_vit, device_ids=device_ids).to('cuda:0')

        if self.pretrain_WSI == 'None':
            self.global_phi = self.global_phi.to(device)
            self.global_transformer = self.global_transformer.to(device)
            self.global_attn_pool = self.global_attn_pool.to(device)
            self.global_rho = self.global_rho.to(device)

        self.classifier = self.classifier.to(device)
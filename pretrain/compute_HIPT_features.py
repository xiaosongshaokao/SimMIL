import torch
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from utils.hipt_model_utils import get_vit256, get_vit4k
import imageio.v2 as imageio
from torchvision.datasets.folder import ImageFolder
from PIL import Image
import os, glob, sys
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import pandas as pd

class HIPT_4K(torch.nn.Module):
    """
    HIPT Model (ViT_4K-256) for encoding non-square images (with [256 x 256] patch tokens), with 
    [256 x 256] patch tokens encoded via ViT_256-16 using [16 x 16] patch tokens.
    """
    def __init__(self, 
        model4k_path,
        device4k=torch.device('cuda:0')):

        super().__init__()
        self.model256 = get_vit4k(pretrained_weights=model4k_path).to(device4k)
        self.model256.head = torch.nn.Identity()
        self.device256 = device4k
	
    def forward(self, x):
        
        return self.model256(x)                     
    

class BagDataset():
    def __init__(self, csv_file, transform = None):
        self.file_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path = self.file_list[idx]
        img = np.load(path)
        # img = pd.read_csv(path, index_col=0)
        # img.reset_index(drop=True, inplace=True)
        # img = img.values
        if self.transform:
            img = self.transform(img)
        img = img.transpose(0, 1)
        # img = img.unfold(1, 16, 16).transpose(1,2).squeeze(dim=0) # 384, 16, 16 
        # img = img.transpose(2, 0, 1) # 384, 32, 32
        sample = {'input' : img}

        return sample

def build_dataset(images_path):
    transformed_dataset = BagDataset(csv_file=images_path, 
                                     transform=transforms.Compose(
                                         [transforms.ToTensor()]
                                     ))
    
    dataloader = DataLoader(transformed_dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False)

    return dataloader


def run(bag_list, model, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.eval()
    num_bags = len(bag_list)
    for i in range(0, num_bags):
        WSI_name = bag_list[i].split(' ')[0]
        feats_list = []
        images_path = glob.glob(os.path.join('/remote-home/share/songyicheng/HIPT_datasets/tcga_4096_feats/', WSI_name, '*.npy'))
        dataloader = build_dataset(images_path=images_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                images = batch['input'].float().cuda()  # bz x 384 x 16 x 16

                feats = model.forward(images).cpu().numpy() # bz x 192
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)))
        if len(feats_list) == 0:
            print('no feats')
        else:
            df = pd.DataFrame(feats_list)
            if not os.path.exists(os.path.join(save_path, 'tcga_4096')):
                os.mkdir(os.path.join(save_path, 'tcga_4096'))
            df.to_csv(os.path.join(save_path, 'tcga_4096', WSI_name + '.csv'), index=False, float_format='%.4f')
            
    return 0

if __name__ == '__main__':
    # input 一个模型，一系列保存成csv的bag，一个保存的路径
    # output 类比dsmil中的compute文件，将一个bag中的所有patch的特征保存成一个csv文件
    dataset = '4ktcga_HIPT_drop_finetune1'
    model = HIPT_4K(model4k_path='/remote-home/share/GraphMIL/backbone_check/debug/finetune/HIPT_TCGA_epoch1/feature_extract.pth')
    model.eval()
    # '/remote-home/share/GraphMIL/backbone_check/debug/HIPT_TCGA/feature_extract_05.pth'
    # '/remote-home/kongyijian/GraphMIL/backbone_aggregation/pretrain_models/HIPT/vit4k_xs_dino.pth'
    bags_path = '/remote-home/kongyijian/MIL/SimMIL/data/TCGA_4096/tcga_4096.csv'
    bag_list = list(pd.read_csv(bags_path)['0'])
    save_path = os.path.join('/remote-home/share/songyicheng/HIPT_datasets/', dataset)

    run(model=model, bag_list=bag_list, save_path=save_path)

    label_path_txt = os.path.join(save_path, dataset + '.txt')
    label_path_csv = os.path.join(save_path, dataset + '.csv')

    with open(label_path_txt, 'w') as f:
        f.write('0 1\n')
        for bag in bag_list:
            f.write(os.path.join(save_path, 'tcga_4096', bag.split(' ')[0]) + ' ' + bag.split(' ')[1] + '\n')
    txt = pd.read_csv(label_path_txt, delimiter=' ')
    txt.to_csv(label_path_csv, index=False, sep=',')



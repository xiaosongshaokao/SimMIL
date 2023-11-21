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
import pandas as pd

class HIPT_4K(torch.nn.Module):
    """
    HIPT Model (ViT_4K-256) for encoding non-square images (with [256 x 256] patch tokens), with 
    [256 x 256] patch tokens encoded via ViT_256-16 using [16 x 16] patch tokens.
    """
    def __init__(self, 
        model256_path: str = '/remote-home/kongyijian/GraphMIL/backbone_aggregation/pretrain_models/HIPT/vit256_small_dino.pth',
        device256=torch.device('cuda:0')):

        super().__init__()
        self.model256 = get_vit256(pretrained_weights=model256_path).to(device256)
        self.device256 = device256
	
    def forward(self, x):
        """
        Forward pass of HIPT (given an image tensor x), outputting the [CLS] token from ViT_4K.
        1. x is center-cropped such that the W / H is divisible by the patch token size in ViT_4K (e.g. - 256 x 256).
        2. x then gets unfolded into a "batch" of [256 x 256] images.
        3. A pretrained ViT_256-16 model extracts the CLS token from each [256 x 256] image in the batch.
        4. These batch-of-features are then reshaped into a 2D feature grid (of width "w_256" and height "h_256".)
        5. This feature grid is then used as the input to ViT_4K-256, outputting [CLS]_4K.

        Args:
          - x (torch.Tensor): [1 x C x W' x H'] image tensor.

        Return:
          - features_cls4k (torch.Tensor): [1 x 192] cls token (d_4k = 192 by default).
        """                                                   # 1. [1 x 3 x W x H].
        
        return self.model256(x)                             # 2. [1 x 384]
    

class BagDataset():
    def __init__(self, csv_file, transform = None):
        self.file_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path = self.file_list[idx]
        img = imageio.imread(path)
        img = Image.fromarray(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        sample = {'input' : img, 'path' : path}

        return sample

def build_dataset(images_path):
    transformed_dataset = BagDataset(csv_file=images_path, 
                                     transform=transforms.Compose(
                                         [transforms.ToTensor()]
                                     ))
    
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=False, num_workers=8, drop_last=False)

    return dataloader, len(transformed_dataset)


def run(bag_list, model, save_path):
    model.eval()
    num_bags = len(bag_list)
    for i in range(0, num_bags):
        WSI_name = bag_list[i].split(' ')[0]
        if not os.path.exists(os.path.join(save_path, WSI_name)):
            os.mkdir(os.path.join(save_path, WSI_name))
            images_path = glob.glob(os.path.join('/remote-home/kongyijian/datasets/tcga_4096', WSI_name, '*.jpg'))
            dataloader, bag_size = build_dataset(images_path=images_path)
            with torch.no_grad():
                for iteration, batch in enumerate(dataloader):
                    images = batch['input'].float().cuda()  # bz x 3 8192 x 8192
                    paths = batch['path']
                    bz = images.shape[0]
                    images = images.unfold(2, 512, 512).unfold(3, 512, 512) # bz x 3 x 16 x 16 x 512 x 512
                    images = images.reshape(bz, -1, 3, 512, 512) # bz x 256 x 3 x 512 x 512

                    for j in range(bz):
                        feature_cls256 = []
                        image = images[j]
                        name = str(paths[j]).split('/')[5]
                        id = str(paths[j]).split('/')[-1].split('.')[0]
                        if os.path.exists(os.path.join(save_path, name, str(id) + '.csv')):
                            continue
                        for mini_bs in range(0, images.shape[1], 64):
                            minibatch_256 = image[mini_bs:mini_bs+64].cuda() # [64, 3, 512, 512]
                            feature_cls256.append(model(minibatch_256).detach().cpu()) # [64, 384]
                        feature_cls256 = torch.cat(feature_cls256, dim=0).numpy() # [256, 384]
                        df = pd.DataFrame(feature_cls256)
                        df.to_csv(os.path.join(save_path, name, str(id) + '.csv'))
                    
                    # all_features = torch.vstack(all_features) # [B, 256, 384]
                    sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)))
        else:
            images_path = glob.glob(os.path.join('/remote-home/kongyijian/datasets/tcga_4096', WSI_name, '*.jpg'))
            saved_csv = os.listdir(os.path.join(save_path, WSI_name))
            if len(saved_csv) == len(images_path):
                continue
            else:
                 for csv in saved_csv:
                    img = pd.read_csv(os.path.join(save_path, WSI_name, csv), index_col=0)
                    img.reset_index(drop=True, inplace=True)
                    img = img.values
                    if img.size == 98304:
                        saved_id = csv.split('.')[0]
                        images_path.remove(os.path.join('/remote-home/kongyijian/datasets/tcga_4096', WSI_name, saved_id + '.jpg'))
                    else:
                        continue
            dataloader, bag_size = build_dataset(images_path=images_path)
            with torch.no_grad():
                for iteration, batch in enumerate(dataloader):
                    images = batch['input'].float().cuda()  # bz x 3 8192 x 8192
                    paths = batch['path']
                    bz = images.shape[0]
                    images = images.unfold(2, 512, 512).unfold(3, 512, 512) # bz x 3 x 32 x 32 x 512 x 512
                    images = images.reshape(bz, -1, 3, 512, 512) # bz x 256 x 3 x 512 x 512

                    for j in range(bz):
                        feature_cls256 = []
                        image = images[j]
                        name = str(paths[j]).split('/')[5]
                        id = str(paths[j]).split('/')[-1].split('.')[0]
                        if os.path.exists(os.path.join(save_path, name, str(id) + '.csv')):
                            continue
                        for mini_bs in range(0, images.shape[1], 64):
                            minibatch_256 = image[mini_bs:mini_bs+64].cuda() # [64, 3, 512, 512]
                            feature_cls256.append(model(minibatch_256).detach().cpu()) # [256, 384]
                        feature_cls256 = torch.cat(feature_cls256, dim=0).numpy() # [256, 384]
                        df = pd.DataFrame(feature_cls256)
                        df.to_csv(os.path.join(save_path, name, str(id) + '.csv'))
                    
                    # all_features = torch.vstack(all_features) # [B, 1024, 384]
                    sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)))

    return 0

if __name__ == '__main__':
    model = HIPT_4K()

    bags_path = '/remote-home/kongyijian/MIL/SimMIL/data/TCGA_4096/tcga_4096.csv'
    bag_list = list(pd.read_csv(bags_path)['0'])
    save_path = '/remote-home/share/songyicheng/HIPT_datasets/tcga_4096_feats/'

    run(model=model, bag_list=bag_list, save_path=save_path)
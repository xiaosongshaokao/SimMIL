import imageio.v2 as imageio
from torchvision.datasets.folder import ImageFolder
from PIL import Image
from PIL import ImageFile
import torch
import numpy as np
import pandas as pd
import os
# ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyImageFolder(ImageFolder):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = []

    def my_loader(self, path):
        img = imageio.imread(path)
            # print(path)
        img = Image.fromarray(img)
        return img.convert('RGB')


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.my_loader(path) 
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return {"data":sample, "label": target}
        # return sample, target

class TCGAImageFolder(ImageFolder):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def my_loader(self, path):
        img = imageio.imread(path)
        img = Image.fromarray(img)
        
        return img.convert('RGB')


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.my_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"data":sample, "label": target}
    
class HIPT_MyImageFolder():
    def __init__(self, csv_file, transform = None):
        self.file_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path, target = self.file_list[idx]
        img = np.load(path)
        # try:
        #     img = pd.read_csv(path, index_col=0)
        # except:
        #     print(path)
        # img.reset_index(drop=True, inplace=True)
        # img = img.values
        if self.transform:
            img = self.transform(img).float()
        img = img.transpose(0, 1) # 384, 16, 16 
        
        sample = {'data' : img, 'label': target}

        return sample



if __name__=="__main__":
    import numpy as np
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    data_root = "/remote-home/share/DATA/NCTCRC/VAL_TEST/test/"
    dataset = MyImageFolder(data_root, train_augmentation)
    # data_root = "/remote-home/source/DATA/NCTCRC/NCT-CRC-HE-100K/"
    # dataset = MyImageFolder(data_root, None)
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True,
            num_workers=0, pin_memory=True)
    for i, data in enumerate(train_loader):
        print(data["data"].shape)
        print(data["label"].shape)
        break
    print(len(dataset))
    print(dataset.class_to_idx)
    # sample_file = "/remote-home/share/GraphMIL/moco/samples/nctcrc/49995_balanced_labels/00.txt"
    # with open(sample_file, "r") as f:
    #     sample_names = [x.split(" ")[0] for x in f.readlines()]
    
    # print(sample_names[:5])

    # path_list = list(filter(lambda x: x[0].rsplit("/")[-1] in sample_names, dataset.samples))

    # print(len(path_list))

    # print(len(dataset.samples))

    # dataset.samples = path_list

    # print(len(dataset))

    

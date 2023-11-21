import os, sys
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from openslide.deepzoom import DeepZoomGenerator
import openslide
from PIL import ImageFilter
import random
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class DebugDataset(Dataset):
    def __init__(self, augmentation, class_num=3):
        self.augmentation = augmentation
        self.image_list = [torch.randn([3,256,256]) for _ in range(512)]
        self.label_list = [random.randint(0,class_num-1) for _ in range(512)]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        data = {}
        data['data'] = self.image_list[idx]
        data['label'] = self.label_list[idx]

        return data


if __name__=="__main__":
    dataset = DebugDataset(None)
    print(dataset[0])
    loader = DataLoader(dataset, batch_size=2)
    loader_iter = iter(loader)
    print(loader_iter.__next__())
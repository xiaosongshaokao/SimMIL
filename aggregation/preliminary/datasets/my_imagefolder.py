import torch
from torchvision.datasets.folder import ImageFolder
import imageio
from PIL import Image
from torchvision import io

class MyImageFolder(ImageFolder):
    
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


# if __name__=="__main__":
#     import numpy as np
#     data_root = "/remote-home/source/DATA/NCTCRC/NCT-CRC-HE-100K/"
#     dataset = MyImageFolder(data_root, None)
#     print(len(dataset))
#     print(dataset.class_to_idx)
#     sample_file = "/remote-home/my/GraphMIL/moco/samples/nctcrc/49995_balanced_labels/00.txt"
#     with open(sample_file, "r") as f:
#         sample_names = [x.split(" ")[0] for x in f.readlines()]
    
#     print(sample_names[:5])

#     path_list = list(filter(lambda x: x[0].rsplit("/")[-1] in sample_names, dataset.samples))

#     print(len(path_list))

#     print(len(dataset.samples))

#     dataset.samples = path_list

#     print(len(dataset))

    

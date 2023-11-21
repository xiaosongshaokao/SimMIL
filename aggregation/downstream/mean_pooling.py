import torch
import torch.nn as nn

class MeanPooling(nn.Module):
    def __init__(self, input_size=512):
        super().__init__()
        self.fc = nn.Linear(input_size, 2)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
    
    def forward(self, feats):
        feats = feats.mean(0)
        feats = self.fc(feats)
        return feats
    
def mean_pooling(input_size=512):
    return MeanPooling(input_size=input_size)
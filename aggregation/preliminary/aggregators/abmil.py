import torch
import torch.nn as nn
import torch.nn.functional as F


class abmilAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 2048 
        self.D = 524  
        self.K = 1
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.L*self.K, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, feats):
        batch_size, bag_size, in_feat = feats.shape
        A = self.attention(feats.view(batch_size * bag_size, in_feat))
        A = A.view(batch_size, bag_size, self.K).softmax(dim=1)
        return torch.bmm(A.permute(0, 2, 1), feats).view(batch_size, in_feat)
    
def abmil():
    return abmilAttention()
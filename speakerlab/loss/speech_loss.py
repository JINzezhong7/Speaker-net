import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *
import numpy

# This function is modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class Speech_Loss(nn.Module):
    def __init__(self,T,t,encoder, **kwargs): # No temp param
        super(Speech_Loss, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(1, T - t + 1), stride=(1, 1))
        # self.fc   = nn.Linear(1536,768)
        ## wav2vec dim = 32; wavlm dim = 31
        self.fc   = nn.Linear(1024,32)

    def forward(self, wav2_fea,ecapa_fea):
        batch_size = wav2_fea.shape[0]
        w1 = F.normalize(wav2_fea, p=2,dim=2)
        e1 = F.normalize(ecapa_fea,p=2,dim=2)
        ## calculate loss
        dot_feature = torch.cosine_similarity(w1,e1, dim = 2)
        dot_feature_sum = dot_feature.mean(dim=1)
        # dot_feature_sum_norm = F.normalize(dot_feature_sum, p=1, dim=0)
        dot_part = torch.ones(dot_feature_sum.shape).cuda() - dot_feature_sum
        l_cos = dot_part.sum(dim=0)/(batch_size)
        return l_cos

    




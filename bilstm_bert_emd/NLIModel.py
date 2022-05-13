import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np


class SiameseNLI(nn.Module):
    def __init__(self,input_size=768,num_layers=1,hidden_size=128):
        super(SiameseNLI, self).__init__()
        
        self.lstm = torch.nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)
            
        self.w1 = torch.nn.Linear(hidden_size*2*2, 128,bias=True)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        self.bn1 = torch.nn.BatchNorm1d(128)
        
        self.w2 = torch.nn.Linear(128, 3,bias=True)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
    

    def forward(self, sent1, sent2):
        sent1 = self.lstm(sent1)
        sent2 = self.lstm(sent2)

        _,seq_size1,_ = sent1[0].size()
        sent1 = sent1[0][:,seq_size1-1,:]
        
        _,seq_size2,_ = sent2[0].size()
        sent2 = sent2[0][:,seq_size2-1,:]

        x = torch.cat((sent1,sent2),1)

        x = torch.relu(self.bn1(self.w1(x)))
        x = self.w2(x)
        return x



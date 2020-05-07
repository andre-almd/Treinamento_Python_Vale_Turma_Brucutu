# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:47:40 2020

@author: André
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#Potholes1
class Net(nn.Module):
    
    # Função inicial que define as camadas da rede
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5) # Saída 220 - Pool 110
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5) # Saída 106 - Pool 53
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=64 * 53 * 53, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=2)

    # Função necessária que define a passagem dos dados pela rede
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 53 * 53)
        
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def layer1(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x
    
    def layer2(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x
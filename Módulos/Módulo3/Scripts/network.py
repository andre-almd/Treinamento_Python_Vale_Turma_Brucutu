# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:53:18 2020

@author: André
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Construir a CNN
# ***************************************************************************************
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        

    def forward(self, x):
        
        x = self.conv1(x) # Saída 24 X 24
        x = F.relu(x)
        x = self.pool(x) # saída 12 X 12
        x = self.conv2(x) # Saída 8 X 8
        x = F.relu(x)
        x = self.pool(x) # saída 4 X 4
        
        x = x.view(-1, 320)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
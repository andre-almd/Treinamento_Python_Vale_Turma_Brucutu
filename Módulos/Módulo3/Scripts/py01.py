# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:47:57 2020

@author: André
"""

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import network as net

# Transformações necessárias ao ler os dados
# ***************************************************************************************
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

# Obter os dados da base do pytorch e salvar os loaders com as informações
# ***************************************************************************************
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=6, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=6, shuffle=False)


# Carregar o modelo salvo
model = net.Net()
model.load_state_dict(torch.load('MNIST_net.pth'))

# Visualizar pesos da rede
for param in model.parameters():
    print(param.data)
  
  
# ************
model.eval()

examples = enumerate(testloader)

batch_idx, (example_data, example_targets) = next(examples)

out1 = model.outConv1(example_data)
out2 = model.outPool1(example_data)
out3 = model.outConv2(example_data)
out4 = model.outPool2(example_data)

# Imagem e filtro
plt.figure('Conv1')
plt.imshow(out1[4][6].detach().numpy(), cmap='gray')

plt.figure('Pool1')
plt.imshow(out2[4][6].detach().numpy(), cmap='gray')

plt.figure('Conv2')
plt.imshow(out3[4][15].detach().numpy(), cmap='gray')

plt.figure('Pool2')
plt.imshow(out4[4][15].detach().numpy(), cmap='gray')

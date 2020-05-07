# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:35:56 2020

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
from torch.autograd import Variable

import cv2

# Transformações necessárias ao ler os dados
# ***************************************************************************************
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

# Carregar o modelo salvo
model = net.Net()
model.load_state_dict(torch.load('MNIST_net.pth'))

# Visualizar pesos da rede
for param in model.parameters():
  print(param.data)
  
# Definir as classes para uso posterior, se necessário
# ***************************************************************************************
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Ler imagem e jogar na rede
img = cv2.imread('imagensDigitos/image14.png', 0)
plt.imshow(img, cmap='gray')

img = transform(img).float()
img = Variable(img, requires_grad=False)
img = img.unsqueeze(0)

output = model(img)
_, pred = torch.max(output, 1)

plt.title(f'Número predito: {classes[pred]}')


# Loop para passar todas as 29 imagens pela rede individualmente e ver a classificação
# ***************************************************************************************
for i in range(1, 29):
    img = cv2.imread('imagensDigitos/image' + str(i) + '.png', 0)
    #plt.figure(i)
    plt.imshow(img, cmap='gray')
    
    img = transform(img).float()
    img = Variable(img, requires_grad=False)
    img = img.unsqueeze(0)

    output = model(img)
    _, pred = torch.max(output, 1)

    plt.title(f'Número predito: {classes[pred]}')

    plt.pause(1.5)
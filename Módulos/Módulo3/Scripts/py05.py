# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:08:40 2020

@author: André
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import py03Network as net


# Transformações nas imagens para processamento:
# Transformar para tensor
# Normalizar os canais RGB com uma média e desvio padrão de 0.5
# Escalar as imagens entre -1 e 1
# *****************************************************************************
data_transforms = {
    'treinamento': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'teste': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# Caminho das pastas de imagens para treinamento e teste
# Mude este caminho de acordo o seu sistema!!!
# *****************************************************************************
data_dir = './datasetPotholes'

# Criação dos conjuntos de dados
# *****************************************************************************

trainset = datasets.ImageFolder(data_dir + r'\treinamento', data_transforms['treinamento'])

testset = datasets.ImageFolder(data_dir + r'\teste', data_transforms['teste'])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True)

# Carregar o modelo salvo
model = net.Net()
model.load_state_dict(torch.load('networkPotholes.pth'))


# Loop para classificar todas as imagens de teste e verificar a taxa de acerto
# *****************************************************************************
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Taxa de classificações corretas para as 60 imagens de teste: {(100 * correct / total)} %')


# Ler um batch e plotar as imagens com predição
# ***************************************************************************************
model.eval()
examples = enumerate(testloader)

batch_idx, (example_data, example_targets) = next(examples)

example_data.shape

outputs = model(example_data)
_, predicted = torch.max(outputs, 1)

fig = plt.figure()
for i in range(5):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  
  # Desnormalizar
  example_data[i] = example_data[i] * 0.5 + 0.5
  
  # [3, 224, 224] -> [224, 224, 3]
  plt.imshow(example_data[i].permute(1, 2, 0))
  
  plt.title(f'Ground Truth: {example_targets[i]} \n Predicted: {[predicted[i].item()]}')
  plt.xticks([])
  plt.yticks([])
fig


# Visualizar Filtros
# **************************************************************
out1 = model.layer1(example_data)
out2 = model.layer2(example_data)

# Imagem e filtro
plt.figure('Layer1')
plt.imshow(out1[0][19].detach().numpy(), cmap='gray')

plt.figure('Layer2')
plt.imshow(out2[0][40].detach().numpy(), cmap='gray')

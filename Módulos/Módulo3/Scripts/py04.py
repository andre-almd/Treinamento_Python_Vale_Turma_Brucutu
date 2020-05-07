# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:55:12 2020

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

# Informações úteis dos conjuntos para serem mostradas no terminal
# *****************************************************************************
trainset_size = len(trainset)
testset_size = len(testset)
print(f'Qt. de imagens de treino: {trainset_size} \nQt. de imagens de teste:{testset_size}')

# Lista com nomes das classes indexadas
class_names = trainset.classes
print(class_names)

# Verificar disponibilidade de GPU - 
# device vai ser o dispositivo para colocar o modelo e os tensores
# Se houver GPu disponível o processamento é realizado nela
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Ler um batch e plotar as imagens
# ***************************************************************************************
examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape

fig = plt.figure()
for i in range(5):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  
  # Desnormalizar
  example_data[i] = example_data[i] * 0.5 + 0.5
  
  # [3, 224, 224] -> [224, 224, 3]
  plt.imshow(example_data[i].permute(1, 2, 0))
  
  plt.title(f'Ground Truth: {example_targets[i]}')
  plt.xticks([])
  plt.yticks([])
fig

# Mostra shape de uma imagem
example_data[0].shape

# Criação do objeto da rede
# *****************************************************************************
model = net.Net()

# Jogar a rede para o device reconhecido
model = model.to(device)

# Criação dos critérios de treinamento: Função de erro e otimizador
# Erro: CrossEntropyLoss
# Otimizados: stochastic gradient descent
# *****************************************************************************
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loop de treinamento da rede 
# (Poder ser encapsulado em uma função!)
# *****************************************************************************
for epoch in range(35):  # loop over the dataset multiple times (Épocas)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # Jogar os dados para o device reconhecido
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 0:    # print every 5 mini-batches
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 5 :.5f}')
            running_loss = 0.0

print('Finished Training')

# Etapa para salvar o modelo da rede treinanda
# Mude este caminho de acordo o seu sistema!!!
# *****************************************************************************
PATH = './networkPotholes.pth'
torch.save(model.state_dict(), PATH)

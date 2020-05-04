# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:38:13 2020

@author: André
"""

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Dataset : MNIST

# Transformações necessárias para aplicar aos dados de entrada
# Transforma em tensor e normaliza com base nas informações de média e desvio do MNIST
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


# Definir as classes para uso posterior, se necessário
# ***************************************************************************************
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9') 


# Ler um batch e plotar as imagens
# ***************************************************************************************
examples = enumerate(testloader)

batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray')
  plt.title(f'Ground Truth: {example_targets[i]}')
  plt.xticks([])
  plt.yticks([])
fig


# Construir a CNN
# ***************************************************************************************
class Net(nn.Module):
    
    # Método padrão para criar a estrutura na chamada da classe
    def __init__(self):
        super(Net, self).__init__()
        
        # Dado de entrada 28X28
        
        # Camadas desejadas de convolução
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        
        # Camada de pooling
        self.pool = nn.MaxPool2d(2,2)
        
        # Camadas totalmente conectadas (ANN) para classificação
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        

    # Método padrão que define a sequeência de passagem (forward) da imagem na rede
    def forward(self, x):
        
        # Dado de entrada 28X28
        
        # Primeira conv com relu e pooling
        x = self.conv1(x) # Saída 24 X 24
        x = F.relu(x)
        x = self.pool(x) # saída 12 X 12
        
        # Segunda conv com relu e pooling
        x = self.conv2(x) # Saída 8 X 8
        x = F.relu(x)
        x = self.pool(x) # saída 4 X 4
        
        # Linearizar as imagens processadas para a ANN ler esses dados como entrada
        x = x.view(-1, 320)
        
        # Passagem na camada intermediária e de saída
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


# Criar objeto da rede
# ***************************************************************************************
network = Net()


# Criar parâmetros de treino (critério de erro e otimizador dos pesos)
# ***************************************************************************************
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

# Loop para treinar a rede
# ***************************************************************************************

for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'Época: {epoch + 1} , Mini-Batch: {i + 1} , loss: {running_loss / 200}')
            running_loss = 0.0

print('Finished Training')

# Ler um batch de teste e plotar as imagens com o ground Truth e a predição
# ***************************************************************************************
network.eval()

examples = enumerate(testloader)

batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

outputs = network(example_data)
_, predicted = torch.max(outputs, 1)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title(f'Ground Truth: {example_targets[i]} \n Predicted: {classes[predicted[i]]}')
  plt.xticks([])
  plt.yticks([])
fig


# Ler todas as imagens e calcular a acurácia
# ***************************************************************************************
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = network(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct  += (predicted == labels).sum().item()
        
print(f'Accuracy of the network on the 10000 test images: {(100 * correct / total)}')


# Salvar o modelo
# ***************************************************************************************
path = './MNIST_net.pth'
torch.save(network.state_dict(), path)

# Visualizar pesos da rede
for param in network.parameters():
    print(param.data)
  
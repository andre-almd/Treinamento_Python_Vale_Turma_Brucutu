# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:03:31 2020

@author: André
"""

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# ler o arquivo da imagem e plotar
pic = Image.open('arquivos/00-cachorro.jpg')
plt.imshow(pic)
type(pic)

# Converte para array
pic_arr = np.asarray(pic)
type(pic_arr)
plt.imshow(pic_arr)

# Zerar as contribuições dos canais GB
pic_red = pic_arr.copy()
plt.figure('vermelho')
plt.imshow(pic_red)
# R G B

#Valores dos canais individuais
plt.figure('Canal vermelho')
plt.imshow(pic_red[:,:,0], cmap='gray')

#Valores do canal Verde 0-255 
plt.figure('Canal Verde')
plt.imshow(pic_red[:,:,1], cmap='gray')

#Valores do canal Azul 0-255 
plt.figure('Canal Azul')
plt.imshow(pic_red[:,:,2], cmap='gray')

# Zerar os canais Verde e Azul
pic_red[:,:,1] = 0
plt.imshow(pic_red)

pic_red[:,:,2] = 0
plt.imshow(pic_red)

# Fatiar a imagem em partes específicas. 
plt.imshow(pic_arr[800:1250, 800:1450, :])

# Mistura: 0:transparente - 1:opaco
plt.imshow(pic_arr[:, :, :], alpha=0.4)

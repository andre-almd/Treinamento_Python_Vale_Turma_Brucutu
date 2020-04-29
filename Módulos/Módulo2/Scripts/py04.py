# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:51:09 2020

@author: André
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

# Ler imagem - trabalhar com float32, range de 0-1
img = cv2.imread('arquivos/bricks.jpg').astype(np.float32) / 255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# Escrever texto na imagem
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='bricks', org=(10,600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)
plt.figure('img')
plt.imshow(img)

# filtro com convolução: Ex. Média dos pixels
kernel1 = np.ones(shape=(7,7), dtype=np.float32)/16

result = cv2.filter2D(img, -1, kernel1)
plt.figure('result')
plt.imshow(result)

# Suavização com blur e gauss
img = cv2.imread('arquivos/estrada.jpg').astype(np.float32) / 255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure('img')
plt.imshow(img)

#Blur pela média dos pixels sob o filtro: substitui o pixel central
blur = cv2.blur(img, (60,60))
plt.figure('blur')
plt.imshow(blur)

# Suavização pelo filtro gaussiano (Ímpar)
gaussian = cv2.GaussianBlur(img, (61,61), 0)
plt.figure('gaussian')
plt.imshow(gaussian)

# Suavização pela mediana (Ímpar)
median = cv2.medianBlur(img, 5)
plt.figure('median')
plt.imshow(median)

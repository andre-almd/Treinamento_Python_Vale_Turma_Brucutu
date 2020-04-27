# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:28:05 2020

@author: André
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

# Espaço de cores 
# (Correção do que ocorreu na aula... lá fiz uma transformação para RGB que passou despercebido. 
#           Agora está tudo ok!)
# *****************************************************************************
# BGR
img = cv2.imread('arquivos/00-cachorro.jpg')
plt.imshow(img) # Sem transformar para RGB

# BGR para HSL
imgHSL = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
plt.imshow(imgHSL)

# BGR para HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(imgHSV)

# Mudar o Hue (matiz) do espaço HSV
# No openCV H[0 - 179], S[0-255] e V[0-255]
imgHSV2 = imgHSV.copy()
imgHSV2[800:1250, 500:1450, 0] = 179

# Mostrar no espaço RGB (Observar o vermelho leve no meio da foto)
imgHSV2 = cv2.cvtColor(imgHSV2, cv2.COLOR_HSV2RGB)
plt.imshow(imgHSV2)


# Threshold
# *****************************************************************************
'''
# Interessante em aplicações onde os formatos e bordas são mais interessantes.
# Segmenta a imagem em diferentes partes (Preto e branco)
# 
'''

# Ler imagem em escala de cinza
img = cv2.imread('arquivos/rainbow.jpg', 0)
plt.imshow(img, cmap='gray')

# Threshold binário ou binário-inverso
_, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
plt.imshow(th1, cmap='gray')


# Threshold trunc
_, th2 = cv2.threshold(img, 170, 255, cv2.THRESH_TRUNC)
plt.imshow(th2, cmap='gray')

#Treshold adaptativo
img = cv2.imread('arquivos/crossword.jpg', 0)
plt.imshow(img, cmap='gray')

th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
plt.imshow(th4, cmap='gray')


# Histogramas por canais BGR
# ********************************************************************************
horse = cv2.imread('arquivos/horse.jpg')
horseRGB = cv2.cvtColor(horse, cv2.COLOR_BGR2RGB)
plt.imshow(horseRGB)

brick = cv2.imread('arquivos/bricks.jpg')
brickRGB = cv2.cvtColor(brick, cv2.COLOR_BGR2RGB)
plt.imshow(brickRGB)

hist_values = cv2.calcHist([horse], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
hist_values.shape
plt.plot(hist_values)

# Plotar o hist dos três canais
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    hist_values = cv2.calcHist([brick], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(hist_values, color=col)
    plt.xlim([0,256])
    #plt.ylim([0, 50000])
plt.title('Histograma')
plt.show()


# Blurring and smoothing (Desfocagem e suavização)
# *****************************************************************************
'''
# Útil em situações para tirar ruídos ou em aplicações para focar em detalhes gerais
# Muito usado combinado com detecção de bordas
# Sem desfocagem e suavização os algoritmos reconhecem muitas bordas desnecessárias. 
'''
# Ler imagem em escala de cinza - trabalhar com floar32, range de 0-1
img = cv2.imread('arquivos/bricks.jpg').astype(np.float32) / 255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# Correção de Gamma: Aumenta ou diminiu o brilho da imagem
gamma = 1.5

result = np.power(img, gamma)
plt.imshow(result)


# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:44:18 2020

@author: André
"""

import matplotlib.pyplot as plt

import cv2

# Códigos com OpenCV

# Caminho errado
img = cv2.imread('arquivos/00-cachoro.jpg')
type(img)

# Caminho certo
img = cv2.imread('arquivos/00-cachorro.jpg')
type(img)
img.shape

# plotar imagem RGB
# ****************************

#cv2 - Channels BGR
plt.imshow(img)
cv2.imshow('Tela', img)

# Transformar canais
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# Ler diretamente em escala de cinza
img_gray = cv2.imread('arquivos/00-cachorro.jpg', cv2.IMREAD_GRAYSCALE) # Ou 0 como segundo parâmetro
plt.imshow(img_gray, cmap='gray')

img_gray.max()

# Outro mapa de cor
plt.imshow(img_gray, cmap='magma')

# ***************************************************

# Mudar tamanho (Observar que a dimensão é ao contrário do que indica o shape) W,H
new_img = cv2.resize(img, (1000,400))
plt.imshow(new_img)

# Mudar tamanho por escala de W,H
w_ratio = 0.5
h_ratio = 0.5

new_img = cv2.resize(img, (0,0), img, w_ratio, h_ratio)
plt.imshow(new_img)

# Flip
new_img3 = cv2.flip(img, -1)
plt.imshow(new_img3)

# Salvar imagem
plt.imsave('novaImagem.jpg', new_img3)

'''
fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111)
ax.imshow(img)
'''

# Modificar imagem com loop for (Observar plot com cv2)
# ********************************************************************************

# Mudar cores dos pixels
imagem = cv2.imread('arquivos/estrada.jpg')
cv2.namedWindow('Imagem', flags=cv2.WINDOW_NORMAL)

for y in range(0, imagem.shape[0], 20):
    for x in range(0, imagem.shape[1], 20):
        imagem[y:y+5, x:x+5] = (0,255,255)
        
cv2.imshow('Imagem', imagem)
#cv2.waitKey(1000)
#cv2.destroyAllWindows()
# **********************************************************

imagem = cv2.imread('arquivos/estrada.jpg')
cv2.namedWindow('Imagem', flags=cv2.WINDOW_NORMAL)

for y in range(0, imagem.shape[0]):
    for x in range(0, imagem.shape[1]):
        if y < x:
            imagem[y, x] = (0,255,255)
cv2.imshow('Imagem', imagem)


# **********************************************************

imagem = cv2.imread('arquivos/estrada.jpg')
cv2.namedWindow('Imagem', flags=cv2.WINDOW_NORMAL)

for y in range(0, imagem.shape[0], 100):
    for x in range(0, imagem.shape[1], 100):
        imagem = cv2.circle(imagem, center=(x,y), radius=30, color=(255,0,0), thickness=-1)
cv2.imshow('Imagem', imagem)
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:45:31 2020

@author: André
"""

import cv2
import numpy as np

# Segementação da cor

cap = cv2.VideoCapture(0)

# Loop para capturar frames
while True:
    
    # Ler o frame
    ret, frame = cap.read()
    
    # Converter de BGR para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir o range das cores
    lower_color = np.array([0, 100, 100])
    upper_color = np.array([10, 255, 255])
    
    # Threshold da imagem dentro dos limites minimo e máximo
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Display o resultado do frame (após as alterações, se tiver)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('resultado', res)
    
    # Opção para fechar a janela com o teclado
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
# Liberar a câmera e destruir todas as janelas abertas
cap.release()
cv2.destroyAllWindows()

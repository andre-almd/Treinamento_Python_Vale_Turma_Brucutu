# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:53:18 2020

@author: André
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

count = 0
# Loop para capturar frames
while True:
    
    # Ler o frame
    ret, frame = cap.read()
    
    # Desenhar retângulos (Outras opções: Linhas, círculos, polígonos)
    if count >=0 and count <=10:
        frame = cv2.rectangle(frame, (10,450), (50,475), color=(255,0,0), thickness=-1)
        count += 1
        print(count)
        
    if count >10 and count <=20:
        frame = cv2.rectangle(frame, (100,45), (150,75), color=(0,255,0), thickness=-1)
        count += 1
        print(count)
        
    if count >20 and count <=30:
        frame = cv2.rectangle(frame, (500,200), (600,275), color=(0,0,255), thickness=-1)
        count += 1
        print(count)
        
    if count >30:
        count = 0
        
        
    #Desfoque dentro do frame
    frame = cv2.blur(frame, (20,20))
    #frame[20:40, 120:300] = cv2.blur(frame[20:40, 120:300], (30,30))
    #frame[150:300, 120:300, :] = cv2.GaussianBlur(frame[150:300, 120:300, :], (21,21), 0)
    #frame[380:460, 120:300, :] = cv2.GaussianBlur(frame[380:460, 120:300, :], (51,51), 0)
    
    #Gradientes dentro do frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Gradiente X
    frame = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    
    #Gradiente Y
    #frame = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    
    # Display o resultado do frame (após as alterações, se tiver)
    cv2.imshow('frame', frame)
    
    # Opção para fechar a janela com o teclado
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
# Liberar a câmera e destruir todas as janelas abertas
cap.release()
cv2.destroyAllWindows()
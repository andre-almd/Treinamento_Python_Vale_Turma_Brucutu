# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:27:59 2020

@author: André
"""

import cv2
import numpy as np

#Objetos de classificação
face_cascade = cv2.CascadeClassifier('arquivos/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('arquivos/haarcascade_eye.xml')

# *********************************************************************************************
def detect_face(img):
    
  
    face_img = img.copy()
    
    face_rects = face_cascade.detectMultiScale(face_img)
    #face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.1, minNeighbors=5) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,0,255), 4) 
        
    return face_img

# *********************************************************************************************
def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    #eyes = eye_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=3) 
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,255,0), 2) 
        
    return face_img

# *********************************************************************************************
    
cap = cv2.VideoCapture(0)

# Loop para capturar frames
while True:
    
    # Ler o frame
    ret, frame = cap.read()
    
    frame = detect_face(frame)
    frame = detect_eyes(frame)

    
    # Display o resultado do frame (após as alterações, se tiver)
    cv2.imshow('frame', frame)
    
    # Opção para fechar a janela com o teclado
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
# Liberar a câmera e destruir todas as janelas abertas
cap.release()
cv2.destroyAllWindows()

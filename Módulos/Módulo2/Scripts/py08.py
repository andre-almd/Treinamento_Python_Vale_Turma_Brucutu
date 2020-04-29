# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:03:33 2020

@author: André
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt

# Ler imagem
rosto = cv2.imread('arquivos/rosto05.jpg')
rosto = cv2.cvtColor(rosto, cv2.COLOR_BGR2RGB)
#rosto = cv2.GaussianBlur(rosto,(51,33),0)
plt.imshow(rosto)

face_cascade = cv2.CascadeClassifier('arquivos/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('arquivos/haarcascade_eye.xml')


# Função que detecta os rostos
#******************************************************************************************

def detect_face(img):
    
  
    face_img = img.copy()
    
    #face_rects = face_cascade.detectMultiScale(face_img)
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.5, minNeighbors=5)
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,0,255), 4) 
        
    return face_img

#******************************************************************************************


# Função que detecta os olhos
#******************************************************************************************
def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,255,0), 3) 
        
    return face_img

#******************************************************************************************
    
result = detect_face(rosto)

result = detect_eyes(result)

plt.imshow(result)

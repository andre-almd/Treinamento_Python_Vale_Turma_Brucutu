# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 20:08:28 2020

@author: André
"""

import cv2
import time

# Conectar com a câmera - 0:câmera padrão
cap = cv2.VideoCapture(0)

# Capturar informações de largura e altura
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Capturar fps da câmera ?????
fps = cap.get(cv2.CAP_PROP_FPS)

# Loop para capturar frames
while True:
    
    # Ler o frame
    ret, frame = cap.read()
    
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame, text='Meu texto!!!', org=(10,400), fontFace=font, 
                fontScale=2, color=(255,0,0), thickness=4)
    
    # Display o resultado do frame (após as alterações, se tiver)
    cv2.imshow('frame', frame)
    
    # Opção para fechar a janela com o teclado
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
# Liberar a câmera e destruir todas as janelas abertas
cap.release()
cv2.destroyAllWindows()





# Contar FPS de fato
# ***********************************************************************************
num_frames = 120
print(f'Capturando {num_frames} frames')

cap = cv2.VideoCapture(0)

# Tempo de inicio
start = time.time()

# Loop para capturar frames
for i in range(0, num_frames):
    ret, frame = cap.read()

# Fim do tempo
end = time.time()

# Tempo que passou
seconds = end - start

print(f'Tempo de captura: {seconds} s')

fps = num_frames / seconds

print(f'FPS estimado: {fps}')

# Liberar a câmera e destruir todas as janelas abertas
cap.release()
cv2.destroyAllWindows()
print('ok!')

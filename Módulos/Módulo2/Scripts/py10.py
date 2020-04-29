# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:48:16 2020

@author: André
"""

import cv2

# Código com opçõa de salvar o vídeo!

# Conectar com a câmera - 0:câmera padrão
cap = cv2.VideoCapture(0)

# Capturar informações de largura e altura
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Capturar fps padrão da câmera
fps = cap.get(cv2.CAP_PROP_FPS)

'''
# Objeto que vai ser responsável por salvar os frames dentro do loop
# Aqui é preciso indicar onde salvar o vídeo e o codec
'''
writer = cv2.VideoWriter('meuVideoSalvo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

# Loop para capturar frames
while True:
    
    # Ler o frame
    ret, frame = cap.read()
    
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame, text='Meu video gravado!', org=(10,400), fontFace=font, 
                fontScale=1, color=(255,0,0), thickness=1)
    
    '''
    #Escrever o frame no arquivo de vídeo
    '''
    writer.write(frame)
    
    # Display o resultado do frame (após as alterações, se tiver)
    cv2.imshow('frame', frame)
    
    # Opção para fechar a janela com o teclado
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
# Liberar a câmera e destruir todas as janelas abertas
cap.release()

'''
# Liberar writer também!
'''
writer.release()

cv2.destroyAllWindows()
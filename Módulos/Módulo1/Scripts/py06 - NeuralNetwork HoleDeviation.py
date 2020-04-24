# -*- coding: utf-8 -*-

# Importar as bibliotecas necessárias
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 

# Módulos do keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout


# Ler os dados para o dataframe 
df = pd.read_csv('well_log.csv')

df = df.drop('Unnamed: 0', axis = 1)
df = df.drop('Neuron_Porosity', axis = 1)

# Ver o cabeçalho do dataset
df.head() 

#Correlation
corr = df.corr()

# Mostrar os valores únicos da classificação 
# 0: Sem desvio
# 1: Com desvio 
classified_label = df['Classification'].unique()
print(f'The labels are: {classified_label}')


# Plotar um gráfico com a contagem
count_Class.plot(kind='bar') 
plt.xlabel('0 : Non-Deviated,  1 : Deviated')
plt.ylabel('Counts')
plt.title('Imbalanced Classification of Hole Deviation')
plt.grid(True) 
plt.show() 

# Especificar as variáveis de entrada)
X = df[['Depth', 'Gamma-ray', 'Shale_Volume', 'Restivity', 
                'Delta T', 'Vs', 'Density', 'Density_Porosity', 'Possions_Ratio']]
X = X.values 

# Especificar a variável de saída
y = df['Classification']
y = y.values

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Normalizar os dados (Tira a media e divide pela variancia)
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)

# Normalizar teste
X_test = scaler.transform(X_test)


# Mostrar o formato dos dados de saida 
print(f'Input Shape: {X.shape}')
print(f'Output Shape: {y.shape}')

# Construir a rede neural
model = Sequential() 
model.add(Dense(12, input_dim = X_train.shape[1], activation = 'sigmoid')) # relu é uma outra função de ativação que pode ser usada aqui
model.add(Dropout(0.1))
model.add(Dense(8, activation = 'sigmoid')) # relu é uma oouyra função de ativação que pode ser usada aqui
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'sigmoid')) # Na saída para classificação relu não é indicada, por conta da transformação que ela insere não se adequar aqui

# Definir os parâmetros do otimizador, do erro e de métrica do modelo
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# Sumário da rede
model.summary() 

# Treinar o modelo - testar 50, 100 e 200 épocas
hist = model.fit(X_train, y_train, epochs = 50, batch_size = 4, validation_data = (scaler.transform(X_test), y_test))

# Avliar o erro e acurácia
accuracy = model.evaluate(scaler.transform(X_test), y_test)[1] * 100 
print(f'The model is {accuracy} Accurate.')

# Plotar a acurácia X epocas
plt.figure(figsize=(20, 8))
plt.grid(True)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Plotar o erro  X epocas
plt.figure(figsize=(20, 8))
plt.grid(True)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show() 

# Lembrando as entradas...
['Depth', 'Gamma-ray', 'Shale_Volume', 'Restivity', 
                'Delta T', 'Vs', 'Density', 'Density_Porosity', 'Possions_Ratio']

# Entrada nova para a rede...
new_input = [[4010, 53, 0.53, 0.55, 125.5, 3450.02, 
            2.2, 0.4, 0.3497]]
# Entrada nova para a rede... 
new_input2 = [[4131, 68.5, 0.7, 0.8, 122.5, 3550.02, 
            2.25, 0.32, 0.395]]

# Reorganizar as novas entradas
new_input = np.array(new_input).reshape(1, -1)
new_input2 = np.array(new_input2).reshape(1, -1)

# Normalizar
new_input = scaler.transform(new_input)
new_input2 = scaler.transform(new_input2)

# Prever e mostrar o resultados para os novos dados
predicted = model.predict_classes(new_input)
probs = model.predict_proba(new_input)

print('The Predicted Classification is : {}'.format(predicted))

# Ler todos os pesos da rede
model.get_weights()

# Ver a configuração da rede
model.get_config()


# Salvar e ler o modelo
model.save('my_model.h5') 

from tensorflow.keras.models import load_model
model1 = load_model('my_model.h5')
probs2 = model1.predict_proba(new_input)

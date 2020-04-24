# -*- coding: utf-8 -*-

# Importar as bibliotecas necessárias

import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import matplotlib.pyplot as plt

# Impirtar Módulo seaborn
import seaborn as sns

# importar o módulo do svm
from sklearn import svm 

# importar módulo de pre processamento para normalizar os dados
from sklearn.preprocessing import StandardScaler 

# Ler os dados para o dataframe 
df = pd.read_csv('well_log.csv')

df = df.drop('Unnamed: 0', axis = 1)
df = df.drop('Neuron_Porosity', axis = 1)

# Ver o cabeçalho do dataset
df.head() 

#Correlation
corr = df.corr()

# Plotar um gráfico com três subplots
fig, (ax1, ax2, ax3) = plt.subplots(1,3)

ax1.plot(df['Gamma-ray'].values, df['Depth'].values, 'b')
ax1.set_title('Gamma ray')

ax2.plot(df['Shale_Volume'].values, df['Depth'].values, 'r')
ax2.set_title('Shale Volume')

ax3.plot(df['Restivity'].values, df['Depth'].values, '#fff555')
ax3.set_title('Resistivity')

plt.show()

# plotar os atributos por pares e observar a linearidade/não linearidade das informações
sns.pairplot(df, hue='Classification')

# Mostrar os valores únicos da classificação 
# 0: Sem desvio
# 1: Com desvio 
classified_label = df['Classification'].unique()
print(f'The labels are: {classified_label}')

# Especificar as variáveis de entrada)
feature_cols = ['Depth', 'Gamma-ray', 'Shale_Volume', 'Restivity', 
                'Delta T', 'Vs', 'Density', 'Density_Porosity', 'Possions_Ratio']

X = df[feature_cols]

# Converter para array
X = X.values 

# Especificar a variável de saída
y = df['Classification']
y = y.values

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Normalizar os dados (Tira a media e divide pela variancia)
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)


#Criar o classificador

# RBF analisa as vizinhanças para determinar em qual grupo a dado está mais relacionado.
# gamma indica quão longe a influência de um único dado chega em relação aos outros
# Pequeno - Longe.   Grande - perto

# C determina se a rigidez da margem. C muito grande margem pequena, mais overfitting. 
# C muito pequeno, margem maior que permite misclassification.

clf1 = svm.SVC(kernel='poly', degree = 2, gamma=0.1, C=1, max_iter=-1) 

clf2 = svm.SVC(kernel='rbf', gamma=0.1, C=1)

# Treinar o classificador
clf1.fit(X_train, y_train)

# LEMBRAR DE NÃO NORMALIZAR PARA VER O ERRO!!!
X_test = scaler.transform(X_test)

# Predizer a resposta para os dados de teste
y_pred = clf1.predict(X_test)

# Acurácia do modelo
# Quão bem meu modelo acertou os dados de teste?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

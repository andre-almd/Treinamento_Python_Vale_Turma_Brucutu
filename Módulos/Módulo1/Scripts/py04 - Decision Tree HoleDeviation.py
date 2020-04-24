# -*- coding: utf-8 -*-

# Importar as bibliotecas necessárias

import pandas as pd 
import matplotlib.pyplot as plt 

# Importar módulo de árvore de decisão do Scikit-learn
import sklearn.tree as tree

# Importar o classificador de árvore de decisão
from sklearn.tree import DecisionTreeClassifier 

# Importar a função para dividir os dados
from sklearn.model_selection import train_test_split

#Importar módulo de métricas 
from sklearn import metrics 

# Ler os dados para o dataframe 
df = pd.read_csv('well_log.csv')

df = df.drop('Unnamed: 0', axis = 1)
df = df.drop('Neuron_Porosity', axis = 1)

# Ver o cabeçalho do dataset
df.head() 

# Correlaçao
corr = df.corr()

# Mostrar os valores únicos da classificação 
# 0: Sem desvio
# 1: Com desvio 
classified_label = df['Classification'].unique()
print(f'Os rótulos são: {classified_label}')

# Contar os rótulos 1 e 0
df['Classification'].value_counts()


# Plotar um gráfico com a contagem 
rotulos_count = df['Classification'].value_counts() 
rotulos_count.plot(kind='bar') 
plt.xlabel('0 : Non-Deviated,  1 : Deviated')
plt.ylabel('Counts')
plt.title('Imbalanced Classification of Hole Deviation')
plt.grid(True) 
plt.show() 

# Especificar as variáveis de entrada)
feature_cols = ['Depth', 'Gamma-ray', 'Shale_Volume', 'Restivity', 
                'Delta T', 'Vs', 'Density', 'Density_Porosity', 'Possions_Ratio']

X = df[feature_cols]

# Converter para array
X = X.values
 
# Especificar a variável de saída
y = df['Classification']
y = y.values

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Criar o objeto do classificador
# Aqui podemos ajustar os parâmetros do classificador para observar as diferenças

# Classificador padrão
clf = DecisionTreeClassifier()

# Classificador com critério de entropia e separação aleatório dos ramos
clf1 = DecisionTreeClassifier(criterion='entropy', splitter='random')

# Classificador com critério de entropia, separação ótima dos ramos e profundidade máxima de 3 níveis
clf2 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3)

# Treinar o classificador (Escolher o clf desejado)
clf = clf.fit(X_train,y_train)

# Predizer a resposta para os dados de teste
y_pred = clf.predict(X_test)

# Acurácia do modelo
# Porcentagem de predições corretas dentro de todas as predições (certas e erradas)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Outras metricas...
# Precisão: Qual a proporcao de identificacoes positivas foi realmente correta?
#           Em outras palavras, o quão bem meu modelo trabalhou.
# Recall: qual proporcao de positivos foi identificados corretamente?
#           quao bom meu modelo e para prever positivos.

clf.feature_importances_

# Mostrar a importancia dos atributos
for feature, importances in zip(feature_cols, clf.feature_importances_):
    print(f'{feature} : {importances}')


# Opçõa de código para pegar os nomes das classes e colocar como parâmetro na função de plot_tree.
# Conversão da lista numérica de classes [0, 1] para texto ['0', '1']
y_train_name = list(clf.classes_)
classes = [str(item) for item in clf.classes_]

# Mostrar a árvore de forma visual
plt.figure(dpi=300)
tree.plot_tree(clf, feature_names=feature_cols, class_names=classes, filled=True,
               proportion=False, rounded=False)

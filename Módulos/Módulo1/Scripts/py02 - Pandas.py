# IMPORTANDO E TRABALHANDO COM DADOS NO PANDAS

import pandas as pd
import numpy as np

# Ler arquivo csv com dados de peso dos estudantes de escolas dos U.S.
dados = pd.read_excel('Students_Weight.xls')

# observar as cinco primeiras linhas do conjunto de dados
primeirosDados = dados.head()

# Acessar colunas
acimaDoPeso = dados[['GRADE LEVEL', 'NO. OVERWEIGHT', 'PCT OVERWEIGHT', 'NO. OBESE', 'PCT OBESE',
                     'NO. OVERWEIGHT OR OBESE', 'PCT OVERWEIGHT OR OBESE', 
                     'SCHOOL YEARS', 'COUNTY', 'CITY', 'STREET ADDRESS', 'ZIP CODE']]

# observar as cinco últimas linhas do conjunto de dados
ultimosDados = dados.tail()

#Dados nulos
acimaDoPeso.isnull()

#Contar dados nulos por coluna
acimaDoPeso[['NO. OVERWEIGHT OR OBESE', 'PCT OVERWEIGHT OR OBESE']].isnull()

acimaDoPeso['NO. OVERWEIGHT OR OBESE'].isnull().value_counts()

# Excluir linhas onde COUNTY é nulo.
acimaDoPeso = acimaDoPeso.dropna(subset=['NO. OVERWEIGHT OR OBESE'])
acimaDoPeso = acimaDoPeso.dropna(subset=['NO. OVERWEIGHT'])
acimaDoPeso = acimaDoPeso.dropna(subset=['NO. OBESE'])

# Criar df menor para ver melhor a substituição dos dados vazios
# *********************************************************************************************************
x = np.random.randn(10,6)*10
df = pd.DataFrame(x, index=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'],
                  columns=['x', 'y', 'z', 'j', 'q', 'k'])

df = df.reindex(['a1', 'a2', 'b1', 'a3', 'a4', 'b2', 'a5', 'a6', 'b3', 
                 'a7', 'a8', 'b4', 'a9', 'a10', 'b5', 'b6'])
df['x'].isnull().value_counts()

# Substituir valorez NA por zero, ou selecionar valor por coluna inserindo um dicionário {'x':0}
df2 = df.fillna(0)

# Substituir propagando o valor para frente
df3 = df.fillna(method='pad')

# Substituir pela média
df4 = df.fillna(df.mean())

# Trabalhar com Correlação: três opções (‘pearson’, ‘kendall’, ‘spearman’)
cor = df4.corr()
# *********************************************************************************************************

# Contar distribuição dos dados
acimaDoPeso['SCHOOL YEARS'].value_counts()
acimaDoPeso['COUNTY'].value_counts()
acimaDoPeso['COUNTY'].value_counts().sort_index()
acimaDoPeso['COUNTY'].unique()

## AGRUPAMENTO - Realizar operações com conjuntos agrupados 
# (Separa os dados, aplica uma função e combina os resultados)

## Importante fazer a pergunta certa...
# Quantas vezes uma máquina parou? Quais instrumentos dão mais problemas em um ano?

group1 = acimaDoPeso.groupby(['SCHOOL YEARS', 'GRADE LEVEL'])

#OPÇÃO DE SELECIONAR COLUNAS 
group2 = acimaDoPeso[['NO. OVERWEIGHT', 'NO. OBESE']].groupby([acimaDoPeso['SCHOOL YEARS'],
                    acimaDoPeso['GRADE LEVEL']]).sum()

group3 = acimaDoPeso[['NO. OVERWEIGHT', 'NO. OBESE']].groupby([acimaDoPeso['SCHOOL YEARS'],
                    acimaDoPeso['GRADE LEVEL']]).mean()

group4 = acimaDoPeso[['NO. OVERWEIGHT', 'NO. OBESE']].groupby([acimaDoPeso['SCHOOL YEARS'],
                    acimaDoPeso['GRADE LEVEL']]).median()

group2.plot.barh(title='Obesos ou acima do peso')
group2[['NO. OVERWEIGHT']].plot.barh(title='Acima do peso')

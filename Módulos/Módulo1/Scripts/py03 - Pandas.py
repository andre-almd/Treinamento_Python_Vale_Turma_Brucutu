# OPERAÇÕES COM STRINGS

import pandas as pd

# Ler arquivo csv com dados de peso dos estudantes de escolas dos U.S.
df = pd.read_excel('Students_Weight.xls')

# Verificar e somar linhas duplicadas
df.duplicated()
df.duplicated().sum()

# observar as dez primeiras linhas do conjunto de dados
df1 = df['AREA NAME'][0:20]

# Extrair os dados por strings
ext1 = df1.str.extract('(RAVENA )')

ext1_Quant = df1.str.extract('(RAVENA)').count()

ext2 = df1.str.extract('(\w+)')

ext3 = df1.str.extract('(\w+)\s(\w+)')

# Separar os dados por opção de texto
ext4 = df1.str.split(' ')
ext4 = df1.str.split(' ', expand=True)

## *** Onde usar: Contagem de palavras, palavras mais citadas em algum serviço.
## *** Quantificar importância de alguns termos.
## *** Mais opções: Verificar fim de palavra, buscar termos específicos, localizar por index,
## ***              juntar termos das colunas, etc.


# FILTRAR OS DADOS
df2 = df['GRADE LEVEL']

df3 = df[df['GRADE LEVEL'] == 'ELEMENTARY']

df4 = df[(df['GRADE LEVEL'] == 'ELEMENTARY') & (df['COUNTY'] == 'ALBANY')]

df5 = df[(df['GRADE LEVEL'] == 'ELEMENTARY') & (df['COUNTY'] == 'ALBANY')][['CITY', 'NO. OBESE']]

df5['CITY'].str.lower()

# substituir Dados
df6 = df['COUNTY'].str.replace('ALBANY', '--')
df4['COUNTY'] = df6
df['COUNTY'] = df6

# Encontrar o tamanho de cada elemento da coluna AREA NAME:
df5['CITY'].str.len()

# Contar quantas escolas cm grau MIDDLE/HIGH tem em Delaware 
highSchool = df[(df['GRADE LEVEL'] == 'MIDDLE/HIGH') & (df['COUNTY'] == 'DELAWARE')]
highSchool['COUNTY'].count()

df['AREA NAME'].str.count('LA ').sum()

# Opçõa de SQL Query para trabalhar no pandas
# Usar a função query para buscar no Data Frame

# Busca simples com função query
df.query('COUNTY == "ALBANY"')

# ### CRIANDO SERIES ###
import pandas as pd

# Serie
serie1 = pd.Series([1, 2, 3, 4, 5, 6])
print(serie1)

# Serie2 com index formatado
serie2 = pd.Series([1, 2, 3, 4, 5, 6], index=['a', 'b', 'c', 'd', 'e', 'f'])
print(serie2)

# ***************************************************************************************************

### CRIANDO DATA FRAMES ###
df1 = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
                   index=['a', 'b', 'c'], columns=['col1', 'col2', 'col3', 'col4', 'col5'])


serie3 = pd.Series([7, 8, 9, 10.1, 11.7])
serie4 = pd.Series([12., 13, 14.7, 15])

# Organizando por linhas
df2 = pd.DataFrame([serie3, serie4])

# Transposta da Serie
serie3.transpose()

# Organizando por colunas
# Criar dicionário para indexar as colunas
dic = {'col1': serie3,
       'col2': serie4}

df3 = pd.DataFrame(dic)
print(df2)

# Listar index e colunas
list(df3.columns)
list(df3.index)

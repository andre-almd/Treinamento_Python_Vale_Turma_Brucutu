# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:24:28 2020

@author: André
"""

# Criando arrays

import numpy as np

# 1 dimensão
arr = np.array([0, 1, 2, 3]).reshape(4,-1)

# 2 dimensões
arr2D = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

# 3 dimensões
arr3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

# Gerar arrays com funções específicas
np.arange(10, 30, 2)
np.zeros((3,3,3))
np.ones((3,3,3))

r = np.random.rand(4, 3, 4)

r.mean(2)
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:36:24 2019

@author: mellis
"""

"""
Plotting to test functions 
"""
import matplotlib.pyplot as plt

dx = 0.005
x = np.arange(-10, 10, dx)
allH = [Ham.create_H3(i) for i in x]

eigProps = [Ham.getEigProps(H, tullyModel) for H in allH]
allE = np.array([i[0] for i in eigProps])
allU = np.array([i[1] for i in eigProps])

allNACV = np.array([Ham.calcNACV(i, dx, tullyModel) for i in x])

plt.plot(x, allU[:, 0, 0])
plt.plot(x, allU[:, 0, 1])
plt.plot(x, allU[:, 1, 0])
plt.plot(x, allU[:, 1, 1])
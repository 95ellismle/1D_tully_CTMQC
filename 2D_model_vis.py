#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:18:13 2019

@author: mellis
"""

import numpy as np
import mayavi.mlab as mlab


x, y = np.mgrid[-10:10:99j, -10:10:99j]

def create_H1(x, y, A=0.01, B=1.6, C=0.005, D=1.0):
    """
    Will create the Hamiltonian in Tully Model 1
    """
    idx = np.sqrt(x**2 + y**2)
    if x > 0 and y > 0:
        V11 = A*(1 - np.exp(-B*idx))
    elif x < 0 and y < 0:
        V11 = -A*(1 - np.exp(B*idx))
    else:
        V11 = 0

    V22 = -V11

    V12 = C * np.exp(-D*(idx**2))

    return np.matrix([[V11, V12], [V12, V22]])

allH = np.zeros((x.shape[0], x.shape[1], 2, 2))
allE = np.zeros((x.shape[0], x.shape[1], 2))
allU = np.zeros_like(allH)
for ix in range(x.shape[0]):
    for iy in range(x.shape[1]):
        H = create_H1(x[ix, iy], y[ix, iy])
        E, U = np.linalg.eigh(H)
        
        allH[ix, iy] = H
        allE[ix, iy] = E
        allU[ix, iy] = U


mlab.mesh(x, y, allE[:, :, 0]*100, color=(1, 0, 0))
#mlab.points3d(x, y, allE[:, :, 1], color=(0, 0, 1))
mlab.show()

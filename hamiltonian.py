#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:34 2019

@author: mellis
"""

import numpy as np


def create_H1(x, A=0.01, B=1.6, C=0.005, D=1.0):
    """
    Will create the Hamiltonian in Tully Model 1
    """
    if x > 0:
        V11 = A*(1 - np.exp(-B*x))
    elif x < 0:
        V11 = -A*(1 - np.exp(B*x))
    else:
        V11 = 0

    V22 = -V11

    V12 = C * np.exp(-D*(x**2))

    return np.matrix([[V11, V12], [V12, V22]])


def create_H2(x, A=0.1, B=0.28, C=0.015, D=0.06, E0=0.05):
    """
    Will create the Hamiltonian in Tully Model 2
    """
    V11 = 0
    V22 = -A * np.exp(-B*(x**2)) + E0
    V12 = C*np.exp(-D*(x**2))

    return np.matrix([[V11, V12], [V12, V22]])


def create_H3(x, A=6e-4, B=0.1, C=0.9):
    """
    Will create the Hamiltonian in Tully Model 3
    """
    V11 = A
    if x < 0:
        V12 = B*np.exp(C*x)
    elif x > 0:
        V12 = B*(2-np.exp(-C*x))
    else:
        V12 = B
    V22 = -A

    return np.matrix([[V11, V12], [V12, V22]])


def create_H4(x, A=6e-4, B=0.1, C=0.9, D=4):
    """
    Will create the Hamiltonian in Tully Model 3
    """
    V11 = A
    V22 = -A
    if x < -D:
        V12 = -B *np.exp(C *(x-D)) + B *np.exp(C *(x+D))
    elif x > D:
        V12 = B *np.exp(-C *(x-D)) - B *np.exp(-C *(x+D))
    else:
        V12 = 2*B - B*np.exp(C *(x-D)) - B *np.exp(-C *(x+D))

    return np.matrix([[V11, V12], [V12, V22]])


def getEigProps(H, ctmqc_env):
    """
    Wrapper function, this really needs taking out it when I have time.
    """
    return np.linalg.eigh(H)


def calcNACVgradPhi(pos, ctmqc_env):
    """
    Will use a different method to calculate the NACV. This function will
    simply use:
        d = <phil | grad phik>
    """
    dx = 1e-5

    H_xm = ctmqc_env['Hfunc'](pos - dx)
    H_x = ctmqc_env['Hfunc'](pos)
    H_xp = ctmqc_env['Hfunc'](pos + dx)

    allU = [getEigProps(H, ctmqc_env)[1]
            for H in (H_xm, H_x, H_xp)]

    gradU = np.gradient(allU, dx, axis=0)
    
    NACV = np.zeros((2, 2))
    for l in range(2):
        for k in range(2):
            NACV[l, k] = np.dot(allU[1][l], gradU[1][k])[0][0]

    return NACV


def calcNACV(irep, ctmqc_env):
    """
    Will calculate the adiabatic NACV for replica irep
    """
    dx = ctmqc_env['dx']
    nState = ctmqc_env['nstate']

    H_xm = ctmqc_env['Hfunc'](ctmqc_env['pos'][irep] - dx)
    H_x = ctmqc_env['H'][irep]
    H_xp = ctmqc_env['Hfunc'](ctmqc_env['pos'][irep] + dx)

    allH = [H_xm, H_x, H_xp]
    gradH = np.gradient(allH, dx, axis=0)[1]
    E, U = getEigProps(H_x, ctmqc_env)
    NACV = np.zeros((nState, nState), dtype=complex)
    for l in range(nState):
        for k in range(nState):
            if l != k:
                phil = np.array(U)[l]
                phik = np.array(U)[k]
                NACV[l, k] = np.dot(phil, np.dot(gradH, phik))
                NACV[l, k] /= E[k] - E[l]

    for l in range(len(NACV)):
        for k in range(l+1, len(NACV)):
            if np.abs(NACV[l, k] + np.conjugate(NACV[k, l])) > 1e-10:
                print("NACV:")
                print(NACV)
                print("NACV[%i, %i]: " % (l, k), NACV[l, k])
                print("NACV[%i, %i]*: " % (l, k), np.conjugate(NACV[k, l]))
                raise SystemExit("NACV not antisymetric!")

    return NACV


def test_Hfunc(Hfunc, minX=-15, maxX=15, stride=0.01):
    allR = np.arange(minX, maxX, stride)
    allH = [Hfunc(x) for x in allR]
    allE = [np.linalg.eigh(H)[0] for H in allH]
    allE = np.array(allE)
    plt.plot(allR, allE[:, 0])
    plt.plot(allR, allE[:, 1])
    return allH, allE

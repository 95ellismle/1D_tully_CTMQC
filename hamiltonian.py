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


def getEigProps(H, ctmqc_env):
    """
    Will return eigen properties (values and vectors) that are usable
    (corrected) minus signs in the code.
    """
    E, U = np.linalg.eigh(H)
    if ctmqc_env['tullyModel'] == 2:
        E1, _ = np.linalg.eig(H)
        if E1[0] > E1[1]:
            U[0, 1] = -U[0, 1]
            U[1, 1] = -U[1, 1]
    return E, U


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


def calcNACV(irep, v, ctmqc_env):
    """
    Will calculate the adiabatic NACV for replica irep
    """
    dx = ctmqc_env['dx']
    nState = ctmqc_env['nstate']

    H_xm = ctmqc_env['Hfunc'](ctmqc_env['pos'][irep, v] - dx)
    H_x = ctmqc_env['H'][irep, v]
    H_xp = ctmqc_env['Hfunc'](ctmqc_env['pos'][irep, v] + dx)

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

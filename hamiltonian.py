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


def calcNACV(irep, ctmqc_env):
    """
    Will calculate the adiabatic NACV for replica irep
    """
    dx = ctmqc_env['dx']

    H_xm = ctmqc_env['Hfunc'](ctmqc_env['pos'][irep] - dx)
    H_x = ctmqc_env['Hfunc'](ctmqc_env['pos'][irep])
    H_xp = ctmqc_env['Hfunc'](ctmqc_env['pos'][irep] + dx)

    allH = [H_xm, H_x, H_xp]
    allU = [getEigProps(H, ctmqc_env)[1] for H in allH]
    grad = np.array(np.gradient(allU, dx, axis=0))[2]
    NACV = np.matmul(allU[1], grad)
    return NACV


def trans_diab_to_adiab(H, u, ctmqc_env):
    """
    Will transform the diabatic coefficients to adiabatic ones
    """
    if len(u) != 2 and len(np.shape(u)) != 1:
        raise SystemExit("Incorrect Shape for diab coeff in trans func")

    U = getEigProps(H, ctmqc_env)[1]

    return np.dot(U.T, u)


def trans_adiab_to_diab(H, C, ctmqc_env):
    """
    Will transform the adiabatic coefficients to diabatic ones
    """
    if len(C) != 2 and len(np.shape(C)) != 1:
        raise SystemExit("Incorrect Shape for diab coeff in trans func")

    U = getEigProps(H, ctmqc_env)[1]

    return np.dot(U, C)

from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:21 2019

@author: mellis
"""
import numpy as np


def calc_ehren_adiab_force(irep, adFrc, adPops, ctmqc_env):
    """
    Will calculate the ehrenfest force in the adiabatic basis
    """
    nstate = ctmqc_env['nstate']
    E = ctmqc_env['E'][irep]
    NACV = ctmqc_env['NACV'][irep]

    # Population Weighted Sum
    F = np.sum(adPops * adFrc)
    
    # NACV bit
    for k in range(nstate):
        for l in range(k):
            if (l==k): continue
            Cl = np.conjugate(ctmqc_env['C'][irep, l])
            Ck = ctmqc_env['C'][irep, k]
            Clk = Cl * Ck
            Ekl = E[k] - E[l]
            F -= 2 * (Clk * Ekl * NACV[l, k]).real
    return F


def calc_QM_force(C, QM, f, ctmqc_env):
    """
    Will calculate the force due to the quantum momentum term for 1 replica and
    1 atom.

    N.B. Doesn't work for multiple atoms at the moment!
    """
    F = 0.0
    for l in range(ctmqc_env['nstate']):
        for k in range(ctmqc_env['nstate']):
            if l == k: continue
            F += f[l] * (f[k] - f[l]) * C[k] * C[l]
        
    F *= -2 * QM

    return F

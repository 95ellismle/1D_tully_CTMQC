#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:21 2019

@author: mellis
"""
import numpy as np

import hamiltonian as Ham


#def calc_ad_frc(pos, ctmqc_env):
#    """
#    Will calculate the forces from each adiabatic state (the grad E term)
#    """
#    dx = ctmqc_env['dx']
#
#    H_xm = ctmqc_env['Hfunc'](pos - dx)
#    H_x = ctmqc_env['Hfunc'](pos)
#    H_xp = ctmqc_env['Hfunc'](pos + dx)
#
#    allH = [H_xm, H_x, H_xp]
#    allE = [np.linalg.eigh(H)[0] for H in allH]
#    grad = np.array(np.gradient(allE, dx, axis=0))[2]
#    return -grad


def calc_ehren_adiab_force(irep, adFrc, adPops, ctmqc_env):
    """
    Will calculate the ehrenfest force in the adiabatic basis
    """
    nstate = ctmqc_env['nstate']
    E = ctmqc_env['E'][irep]
    NACV = ctmqc_env['NACV'][irep]

    F = np.sum(adPops * adFrc)
    for k in range(nstate):
        for l in range(nstate):
            Cl = np.conjugate(ctmqc_env['C'][irep, l])
            Ck = ctmqc_env['C'][irep, k]
            Clk = Cl * Ck
            Ekl = E[k] - E[l]

            F -= Clk * Ekl * NACV[l, k]
    if F.imag > 1e-12:
        raise SystemExit("Something's gone wrong imaginary force!")

    return F.real


def calc_QM_force(adPops, QM, adMom, ctmqc_env):
    """
    Will calculate the force due to the quantum momentum term for 1 replica and
    1 atom.

    N.B. Doesn't work for multiple atoms at the moment!
    """
    F = 0.0
    for l in range(ctmqc_env['nstate']):

        tmp = 0.0
        for k in range(ctmqc_env['nstate']):
            tmp += ((adPops[k] * adMom[k]) - adMom[l])

        F -= (2 * adPops[l] * QM * adMom[l]) * tmp

    return F
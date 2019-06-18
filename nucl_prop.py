#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:21 2019

@author: mellis
"""
import numpy as np

import hamiltonian as Ham


def calc_ehren_adiab_force(irep, v, adFrc, adPops, ctmqc_env):
    """
    Will calculate the ehrenfest force in the adiabatic basis
    """
    nstate = ctmqc_env['nstate']
    ctmqc_env['H'][irep, v] = ctmqc_env['Hfunc'](ctmqc_env['pos'][irep, v])
    E = Ham.getEigProps(ctmqc_env['H'][irep, v], ctmqc_env)[0]
    ctmqc_env['E'][irep, v] = E
    NACV = ctmqc_env['NACV'][irep, v]

    F = np.sum(adPops * adFrc)
    for k in range(nstate):
        for l in range(nstate):
            Cl = np.conjugate(ctmqc_env['C'][irep, v, l])
            Ck = ctmqc_env['C'][irep, v, k]
            Clk = Cl * Ck
            Ekl = E[k] - E[l]
            F -= Clk * Ekl * NACV[l, k]
#            print(NACV[l, k])
    if F.imag > 1e-12:
        msg = "Something's gone wrong ehrenfest force "
        msg += "-it has a imaginary component!"
        raise SystemExit(msg)

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

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:21 2019

@author: mellis
"""
import numpy as np

import hamiltonian as Ham


def calc_ad_frc(pos, ctmqc_env):
    """
    Will calculate the forces from each adiabatic state (the grad E term)
    """
    dx = ctmqc_env['dx']

    H_xm = ctmqc_env['Hfunc'](pos - dx)
    H_x = ctmqc_env['Hfunc'](pos)
    H_xp = ctmqc_env['Hfunc'](pos + dx)

    allH = [H_xm, H_x, H_xp]
    allE = [Ham.getEigProps(H, ctmqc_env)[0] for H in allH]
    grad = np.array(np.gradient(allE, dx, axis=0))[2]
    return grad


def calc_ehren_adiab_force(irep, gradE, adPops, ctmqc_env):
    """
    Will calculate the ehrenfest force in the adiabatic basis
    """
    nstate = ctmqc_env['nstate']
    E = Ham.getEigProps(ctmqc_env['H'][irep], ctmqc_env)[0]
    NACV = Ham.calcNACV(irep, ctmqc_env)

    F = -np.sum(adPops * gradE)
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


def calc_ehren_diab_force(irep, diPops, ctmqc_env):
    """
    Will calculate the force using the diabatic basis
    """
    pass
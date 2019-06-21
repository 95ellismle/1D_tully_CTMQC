#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:10 2019

@author: mellis
"""
import numpy as np

import hamiltonian as Ham


def trans_diab_to_adiab(allH, allu, ctmqc_env):
    """
    Will transform the diabatic coefficients to adiabatic ones
    """
    nrep = ctmqc_env['nrep']
    nstate = ctmqc_env['nstate']
    allC = np.zeros((nrep, nstate), dtype=complex)
    for irep in range(nrep):
        u = allu[irep, :]
        H = allH[irep]
        if len(u) != 2 and len(np.shape(u)) != 1:
            msg = "Incorrect Shape for diab coeff in trans func"
            raise SystemExit(msg)

        U = Ham.getEigProps(H, ctmqc_env)[1]

        allC[irep] = np.dot(np.array(U), np.array(u))
    return allC


def trans_adiab_to_diab(allH, allC, ctmqc_env):
    """
    Will transform the adiabatic coefficients to diabatic ones
    """
    nrep = ctmqc_env['nrep']
    nstate = ctmqc_env['nstate']
    allu = np.zeros((nrep, nstate), dtype=complex)
    for irep in range(nrep):
        C = allC[irep, :]
        H = allH[irep]
        if len(C) != 2 and len(np.shape(C)) != 1:
            msg = "Incorrect Shape for adiab coeff in trans func"
            raise SystemExit(msg)

        U = Ham.getEigProps(H, ctmqc_env)[1]

        allu[irep] = np.dot(np.array(U.T), np.array(C))
    return allu

def calc_ad_pops(C, ctmqc_env=False):
    """
    Will calculate the adiabatic populations of all replicas
    """
    if ctmqc_env is not False:
        nstate = ctmqc_env['nstate']
        if len(C) != nstate or len(np.shape(C)) != 1:
            raise SystemExit("Incorrect Shape for adiab coeff in calc_ad_pops")
    else:
        nstate = len(C)
    pops = np.zeros(nstate)
    for istate in range(nstate):
        pops[istate] = np.linalg.norm(C[istate])
    return pops**2


def renormalise_all_coeffs(coeff):
    """
    Will renormalise all the coefficients for replica I, atom v.
    """
    nrep, nstate = np.shape(coeff)
    norms = np.linalg.norm(coeff, axis=1)
    for I in range(nrep):
        coeff[I, :] = coeff[I, :] / norms[I]

    return coeff


def do_diab_prop_ehren(ctmqc_env):
    """
    Will propagate the coefficients in the diabatic basis (without the
    diabatic NACE)
    """
    for irep in range(ctmqc_env['nrep']):
        dx_E = (ctmqc_env['pos'][irep] - ctmqc_env['pos_tm'][irep])
        dx_E /= float(ctmqc_env['elec_steps'])
    
        X1 = makeX_diab_ehren(ctmqc_env, ctmqc_env['pos_tm'][irep])
        for Estep in range(ctmqc_env['elec_steps']):
            pos2 = ctmqc_env['pos_tm'][irep] + (Estep + 0.5) * dx_E    
            X12 = makeX_diab_ehren(ctmqc_env, pos2)

            pos3 = ctmqc_env['pos_tm'][irep] + (Estep+1) * dx_E
            X2 = makeX_diab_ehren(ctmqc_env, pos3)
    
            ctmqc_env['u'][irep] = __RK4(ctmqc_env,
                                         ctmqc_env['u'][irep], X1, X12, X2)
    
            X1 = X2[:]

def do_adiab_prop_ehren(ctmqc_env, irep):
    """
    Will actually carry out the propagation of the coefficients
    """
    dx_E = (ctmqc_env['pos'] - ctmqc_env['pos_tm'])
    dx_E /= float(ctmqc_env['elec_steps'])

    X1 = makeX_adiab_ehren(ctmqc_env['pos_tm'], irep)
    for Estep in range(ctmqc_env['elec_steps']):
        pos2 = ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
        pos3 = ctmqc_env['pos_tm'] + (Estep+1) * dx_E

        X12 = makeX_adiab_ehren(pos2, irep)
        X2 = makeX_adiab_ehren(pos3, irep)

        ctmqc_env['C'][irep] = __RK4(ctmqc_env,
                                     ctmqc_env['C'][irep], X1, X12, X2)

        X1 = X2[:]

def makeX_diab_ehren(ctmqc_env, pos):
    """
    Will make the diabatic X matrix
    """
    H = ctmqc_env['Hfunc'](pos)

    return -1j * H

def makeX_adiab_ehren(ctmqc_env, pos, irep):
    """
    Will make the adiabatic X matrix
    """
    nstates = ctmqc_env['nstate']
    X = np.zeros((nstates, nstates), dtype=complex)

    H = ctmqc_env['Hfunc'](pos)
    E, U = Ham.getEigProps(H, ctmqc_env)

    NACV = Ham.calcNACV(irep, ctmqc_env)
#        print(NACV, "\n")
    v = ctmqc_env['vel'][irep]

    # First part
    for l in range(nstates):
        X[l, l] = E[l]

    # Second part
    X += -1j * NACV * v[0]

    return -1j * X

def __RK4(ctmqc_env, coeff, X1, X12, X2):
    """
    Will carry out the RK4 algorithm to propagate the coefficients
    """
    dTe = ctmqc_env['dt'] / float(ctmqc_env['elec_steps'])
    
    K1 = np.array(dTe * np.dot(X1, coeff))[0]
    K2 = np.array(dTe * np.dot(X12, coeff + K1/2.))[0]
    K3 = np.array(dTe * np.dot(X12, coeff + K2/2.))[0]
    K4 = np.array(dTe * np.dot(X2, coeff + K3))[0]

    Ktot = (1./6.) * (K1 + (2.*K2) + (2.*K3) + K4)

    coeff = coeff + Ktot

    return coeff

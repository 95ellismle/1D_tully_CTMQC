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

        U = np.linalg.eigh(H)[1]

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

        U = np.linalg.eigh(H)[1]

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


def get_diffVal(var, var_tm, ctmqc_env):
    """
    Will get the necessary variables to do the linear interpolation.
    """
    dvar_E = (var - var_tm) / float(ctmqc_env['elec_steps'])
    return dvar_E


# Diabatic Propagation   
def do_diab_prop_QM(ctmqc_env):
    """
    Will propagate the coefficients in the diabatic basis (without the
    diabatic NACE)
    """
    for irep in range(ctmqc_env['nrep']):
        H_tm = np.matrix(ctmqc_env['H_tm'][irep])
        dH_E = get_diffVal(ctmqc_env['H'][irep], H_tm, ctmqc_env)
        
        U_tm = np.matrix(ctmqc_env['U_tm'][irep])
        dU_E = get_diffVal(ctmqc_env['U'][irep], U_tm, ctmqc_env)
        
        Qlk_tm = np.matrix(ctmqc_env['Qlk_tm'][irep])
        dQlk_E = get_diffVal(ctmqc_env['Qlk'][irep], Qlk_tm, ctmqc_env)
        
        f_tm = np.array(ctmqc_env['adMom_tm'][irep])
        df_E = get_diffVal(ctmqc_env['adMom'][irep], f_tm, ctmqc_env)

        do_QM = np.max(dQlk_E) > 1e-10
        
        if do_QM:
            X1 = makeX_diab_QM(H_tm, Qlk_tm, f_tm, ctmqc_env['C'][irep], U_tm)
        else:
            X1 = makeX_diab_ehren(H_tm)
        for Estep in range(ctmqc_env['elec_steps']):
            H = H_tm + (Estep + 0.5) * dH_E
            if do_QM:
                U = U_tm + (Estep + 0.5) * dU_E
                Qlk = Qlk_tm + (Estep + 0.5) * dQlk_E
                f = f_tm + (Estep + 0.5) * df_E
                X12 = makeX_diab_QM(H, Qlk, f, ctmqc_env['C'][irep], U)
            else:
                X12 = makeX_diab_ehren(H)
                
            H = H_tm + (Estep + 0.5) * dH_E
            if do_QM:
                H = H_tm + (Estep + 1.0) * dH_E
                U = U_tm + (Estep + 1.0) * dU_E
                f = f_tm + (Estep + 1.0) * df_E
                X2 = makeX_diab_QM(H, Qlk, f, ctmqc_env['C'][irep], U)
            else:
                X2 = makeX_diab_ehren(H)

    
            ctmqc_env['u'][irep] = __RK4(ctmqc_env,
                                         ctmqc_env['u'][irep], X1, X12, X2)
            if do_QM:
                ctmqc_env['C'][irep] = np.matmul(np.array(ctmqc_env['U'][irep].T),
                                                 np.array(ctmqc_env['u'][irep]))
            
            X1 = X2[:]


def makeX_diab_QM(H, Qlk, f, C, U):
    """
    Will make the diabatic X matrix
    """    
    C = np.array(C)
    nstates = len(H)
    
    # Get Ehrenfest X
    Xeh = -1j * H

    # Calculate the adiabatic populations
    Cpops = np.conjugate(C) * C

    # ammend X
    Xqm = np.zeros_like(Xeh)
    for l in range(nstates):
        for k in range(nstates):
            Xqm[l, l] += Qlk[l, k] * (f[k] - f[l]) * Cpops[k]
    Xqm = np.matmul(U.T, np.matmul(Xqm, U))
    
    return Xeh + Xqm


def makeX_diab_ehren(H):
    """
    Will make the diabatic X matrix
    """
    return -1j * H


def do_diab_prop_ehren(ctmqc_env):
    """
    Will propagate the coefficients in the diabatic basis (without the
    diabatic NACE)
    """
    for irep in range(ctmqc_env['nrep']):
        H_tm = np.matrix(ctmqc_env['H_tm'][irep])
        dH_E = get_diffVal(ctmqc_env['H'][irep], H_tm, ctmqc_env)

        X1 = makeX_diab_ehren(H_tm)
        for Estep in range(ctmqc_env['elec_steps']):
            H = H_tm + (Estep + 0.5) * dH_E
            X12 = makeX_diab_ehren(H)

            X2 = makeX_diab_ehren(H)
            
            H = H_tm + (Estep + 1.0) * dH_E
            ctmqc_env['u'][irep] = __RK4(ctmqc_env,
                                         ctmqc_env['u'][irep], X1, X12, X2)
    
            X1 = X2[:]


# Adiab Propagation
#def do_adiab_prop_ehren(ctmqc_env):
#    """
#    Will actually carry out the propagation of the coefficients
#    """
#    for irep in range(ctmqc_env['nrep']):
##        H_tm = np.matrix(ctmqc_env['H_tm'][irep])
##        dH_E = get_diffVal(ctmqc_env['H'][irep], H_tm, ctmqc_env)
##        
##        NACV_tm = ctmqc_env['NACV_tm'][irep]
##        dNACV_E = get_diffVal(ctmqc_env['NACV'][irep], NACV_tm, ctmqc_env)
##        pos_tm = np.
#        
#        v_tm = ctmqc_env['vel_tm'][irep]
#        dv_E = get_diffVal(ctmqc_env['vel'][irep], v_tm, ctmqc_env)
#        R_tm = ctmqc_env['pos_tm'][irep]
#        dR_E = get_diffVal(ctmqc_env['pos'][irep], R_tm, ctmqc_env)
#        
#        H = ctmqc_env['Hfunc'](R_tm)
#        NACV = Ham.calcNACV(irep, ctmqc_env)
#        
#        X1 = makeX_adiab_ehren(H, v_tm, NACV)
#        for Estep in range(ctmqc_env['elec_steps']):
##            H = H_tm + (Estep + 0.5) * dH_E
##            NACV = NACV_tm + (Estep + 0.5) * dNACV_E
#            R = R_tm + (Estep + 0.5) * dR_E
#            H = ctmqc_env['Hfunc'](R)
#            H = ctmqc_env['Hfunc'](R)
#            v = v_tm + (Estep + 0.5) * dv_E
#            X12 = makeX_adiab_ehren(H, v_tm, NACV)
#            
#            H = H_tm + (Estep + 1.0) * dH_E
#            NACV = NACV_tm + (Estep + 1.0) * dNACV_E
#            v = v_tm + (Estep + 1.0) * dv_E
#            X2 = makeX_adiab_ehren(H, v, NACV)
#    
#            ctmqc_env['C'][irep] = __RK4(ctmqc_env,
#                                         ctmqc_env['C'][irep], X1, X12, X2)
#    
#            X1 = X2[:]
#
#
#def makeX_adiab_ehren(H, vel, NACV):
#    """
#    Will make the adiabatic X matrix
#    """
#    nstates = len(H)
#    X = np.zeros((nstates, nstates), dtype=complex)
#
#    E, U = np.linalg.eigh(H)
#    
#    # First part
#    X = (-1j * np.identity(2) * E) - (NACV * vel)
#
#    return X


def do_adiab_prop_ehren(ctmqc_env):
    """
    Will actually carry out the propagation of the coefficients
    """
    for irep in range(ctmqc_env['nrep']):
        dx_E = (ctmqc_env['pos'] - ctmqc_env['pos_tm'])
        dx_E /= float(ctmqc_env['elec_steps'])

        ctmqc_env['pos'] = ctmqc_env['pos_tm']
        X1 = makeX_adiab(ctmqc_env, irep)
        for Estep in range(ctmqc_env['elec_steps']):
            ctmqc_env['pos'] = ctmqc_env['pos_tm'] \
                                    + (Estep + 0.5) * dx_E
            X12 = makeX_adiab(ctmqc_env, irep)

            ctmqc_env['pos'] = ctmqc_env['pos_tm'] \
                + (Estep+1) * dx_E
            X2 = makeX_adiab(ctmqc_env, irep)

            coeff = __RK4(ctmqc_env, ctmqc_env['C'][irep], X1, X12, X2)
            ctmqc_env['C'][irep] = coeff

            X1 = X2[:]


def makeX_adiab(ctmqc_env, irep):
    """
    Will make the adiabatic X matrix
    """
    nstates = ctmqc_env['nstate']
    X = np.zeros((nstates, nstates), dtype=complex)
    pos = ctmqc_env['pos']

    H = ctmqc_env['Hfunc'](pos[irep])
    E, U = np.linalg.eigh(H)
    NACV = Ham.calcNACV(irep, ctmqc_env)
    vel = ctmqc_env['vel'][irep]
    
    # First part
    for l in range(nstates):
        X[l, l] = E[l]

    # Second part
    X += -1j * NACV * vel

    return -1j * np.array(X)


def __RK4(ctmqc_env, coeff, X1, X12, X2):
    """
    Will carry out the RK4 algorithm to propagate the coefficients
    """
    dTe = ctmqc_env['dt'] / float(ctmqc_env['elec_steps'])
    
    K1 = np.array(dTe * np.matmul(X1, coeff))[0]
    K2 = np.array(dTe * np.matmul(X12, coeff + K1/2.))[0]
    K3 = np.array(dTe * np.matmul(X12, coeff + K2/2.))[0]
    K4 = np.array(dTe * np.matmul(X2, coeff + K3))[0]

    Ktot = (1./6.) * (K1 + (2.*K2) + (2.*K3) + K4)

    coeff = coeff + Ktot

    return coeff

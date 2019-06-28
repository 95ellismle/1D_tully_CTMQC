#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:10 2019

@author: mellis
"""
import numpy as np
import time

import hamiltonian as Ham
import QM_utils as qUt


def trans_diab_to_adiab(ctmqc_env):
    """
    Will transform the diabatic coefficients to adiabatic ones
    """
    nrep = ctmqc_env['nrep']

    for irep in range(nrep):
        u = ctmqc_env['u'][irep]
        U = ctmqc_env['U'][irep]
        ctmqc_env['C'][irep] = np.matmul(np.array(U.T),
                                         np.array(u))


def trans_adiab_to_diab(ctmqc_env):
    """
    Will transform the adiabatic coefficients to diabatic ones
    """
    nrep = ctmqc_env['nrep']

    for irep in range(nrep):
        C = ctmqc_env['C'][irep]
        U = ctmqc_env['U'][irep]

        ctmqc_env['u'][irep] = np.matmul(np.array(U),
                                         np.array(C))


def renormalise_all_coeffs(coeff):
    """
    Will renormalise all the coefficients for replica I, atom v.
    """
    nrep, _ = np.shape(coeff)
    norms = np.linalg.norm(coeff, axis=1)
    for I in range(nrep):
        coeff[I, :] = coeff[I, :] / norms[I]

    return coeff


def calc_ad_pops(C, ctmqc_env=False):
    """
    Will calculate the adiabatic populations of 1 replica given the coefficient
    C
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


def get_diffVal(var, var_tm, ctmqc_env):
    """
    Will get the necessary variables to do the linear interpolation.
    """
    dvar_E = (var - var_tm) / float(ctmqc_env['elec_steps'])
    return dvar_E


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
            ctmqc_env['u'][irep] = __RK4_2(ctmqc_env,
                                         ctmqc_env['u'][irep], X1, X12, X2)

            X1 = X2[:]


def do_diab_prop_QM(ctmqc_env, irep):
    """
    Will propagate the coefficients in the diabatic basis (without the
    diabatic NACE)
    """
    dx_E = (ctmqc_env['pos'] - ctmqc_env['pos_tm'])
    dx_E /= float(ctmqc_env['elec_steps'])

    X1 = makeX_diab_QM(ctmqc_env, ctmqc_env['pos_tm'], irep)
    for Estep in range(ctmqc_env['elec_steps']):
        pos2 = ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
        pos3 = ctmqc_env['pos_tm'] + (Estep + 1.0) * dx_E

        X12 = makeX_diab_QM(ctmqc_env, pos2, irep)
        X2 = makeX_diab_QM(ctmqc_env, pos3, irep)

        coeff = __RK4(ctmqc_env['u'][irep], X1, X12, X2, ctmqc_env)
        ctmqc_env['u'][irep] = coeff

        X1 = X2[:]


def makeX_diab_QM(ctmqc_env, pos, irep):
    """
    Will make the diabatic X matrix
    """
    nstate = ctmqc_env['nstate']
    H = ctmqc_env['Hfunc'](pos[irep])
    U = Ham.getEigProps(H, ctmqc_env)[1]
    X = -1j * H
    K = np.zeros((nstate, nstate))

    C = trans_diab_to_adiab(H,
                            ctmqc_env['u'][irep],
                            ctmqc_env)
    ctmqc_env['C'][irep] = C
    adPops = calc_ad_pops(C, ctmqc_env)

    ctmqc_env['pos'] = pos
    QM = qUt.calc_QM_analytic(ctmqc_env, irep)
    adMom = qUt.calc_ad_mom(ctmqc_env, irep)
    for l in range(nstate):
        tmp = 0.0
        for k in range(nstate):
            Ck = adPops[k]
            fk = adMom[k]
            tmp += Ck*fk

        K[l, l] = QM * adMom[l] * tmp

    Xctmqc = np.dot(U.T, np.dot(K, U))

    return X - Xctmqc


def do_adiab_prop_ehren(ctmqc_env):
    """
    Will actually carry out the propagation of the coefficients
    """
    for irep in range(ctmqc_env['nrep']):
        v_tm = ctmqc_env['vel_tm'][irep]
        dv_E = get_diffVal(ctmqc_env['vel'][irep], v_tm, ctmqc_env)

        H_tm = np.matrix(ctmqc_env['H_tm'][irep])
        dH_E = get_diffVal(ctmqc_env['H'][irep], H_tm, ctmqc_env)

        E_tm = ctmqc_env['E_tm'][irep]
        dE_E = get_diffVal(ctmqc_env['E'][irep], E_tm, ctmqc_env)

        NACV_tm = np.matrix(ctmqc_env['NACV_tm'][irep])
        dNACV_E = get_diffVal(ctmqc_env['NACV'][irep], NACV_tm, ctmqc_env)

        X1 = makeX_adiab_ehren(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm)
        for Estep in range(ctmqc_env['elec_steps']):
            H_tm = H_tm + (Estep + 0.5) * dH_E
            E_tm = E_tm + (Estep + 0.5) * dE_E
            NACV_tm = NACV_tm + (Estep + 0.5) * dNACV_E
            v_tm = v_tm + (Estep + 0.5) * dv_E
            X12 = makeX_adiab_ehren(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm)

            H_tm = H_tm + (Estep + 1.0) * dH_E
            E_tm = E_tm + (Estep + 1.0) * dE_E
            NACV_tm = NACV_tm + (Estep + 1.0) * dNACV_E
            v_tm = v_tm + (Estep + 1.0) * dv_E
            X2 = makeX_adiab_ehren(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm)

            coeff = __RK4(ctmqc_env['C'][irep], X1, X12, X2,
                          ctmqc_env)
            ctmqc_env['C'][irep] = coeff

            X1 = X2[:]


def makeX_adiab_ehren(ctmqc_env, H, NACV, vel, E):
    """
    Will make the adiabatic X matrix
    """
    return (-1j * np.identity(2) * E) - (NACV * vel)


def do_adiab_prop_QM(ctmqc_env):
    """
    Will actually carry out the propagation of the coefficients
    """
    for irep in range(ctmqc_env['nrep']):
        v_tm = ctmqc_env['vel_tm'][irep]
        dv_E = get_diffVal(ctmqc_env['vel'][irep], v_tm, ctmqc_env)

        H_tm = np.matrix(ctmqc_env['H_tm'][irep])
        dH_E = get_diffVal(ctmqc_env['H'][irep], H_tm, ctmqc_env)

        E_tm = ctmqc_env['E_tm'][irep]
        dE_E = get_diffVal(ctmqc_env['E'][irep], E_tm, ctmqc_env)

        NACV_tm = np.matrix(ctmqc_env['NACV_tm'][irep])
        dNACV_E = get_diffVal(ctmqc_env['NACV'][irep], NACV_tm, ctmqc_env)

        dQlk_E = (ctmqc_env['Qlk'][irep]
                  - ctmqc_env['Qlk_tm'][irep])
        dQlk_E /= float(ctmqc_env['elec_steps'])
        df_E = (ctmqc_env['adMom'][irep]
                - ctmqc_env['adMom_tm'][irep])
        df_E /= float(ctmqc_env['elec_steps'])
        
        # Make X_{1}
        ctmqc_env['Qlk'][irep] = ctmqc_env['Qlk_tm'][irep]
        ctmqc_env['adMom'][irep] = ctmqc_env['adMom_tm'][irep]
        if (irep in ctmqc_env['QM_reps']):
            X1 = makeX_adiab_Qlk(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm, irep)
        else:
            X1 = makeX_adiab_ehren(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm)

        for Estep in range(ctmqc_env['elec_steps']):
            # Linear Interpolationprint(check)
            H_tm = H_tm + (Estep + 0.5) * dH_E
            E_tm = E_tm + (Estep + 0.5) * dE_E
            NACV_tm = NACV_tm + (Estep + 0.5) * dNACV_E
            v_tm = v_tm + (Estep + 0.5) * dv_E
            ctmqc_env['Qlk'][irep] = ctmqc_env['Qlk_tm'][irep]\
                + (Estep + 0.5) * dQlk_E
            ctmqc_env['adMom'][irep] = ctmqc_env['adMom_tm'][irep]\
                + (Estep + 0.5) * df_E

            # Make X12
            if (irep in ctmqc_env['QM_reps']):
                X12 = makeX_adiab_Qlk(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm, irep)
            else:
                X12 = makeX_adiab_ehren(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm)
#            X12 = makeX_adiab_Qlk(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm, irep) 

            # Linear Interpolation
            ctmqc_env['Qlk'][irep] = ctmqc_env['Qlk_tm'][irep]\
                + (Estep + 1.0) * dQlk_E
            ctmqc_env['adMom'][irep] = ctmqc_env['adMom_tm'][irep]\
                + (Estep + 1.0) * df_E
            H_tm = H_tm + (Estep + 1.0) * dH_E
            E_tm = E_tm + (Estep + 1.0) * dE_E
            NACV_tm = NACV_tm + (Estep + 1.0) * dNACV_E
            v_tm = v_tm + (Estep + 1.0) * dv_E

            # Make X2
            if (irep in ctmqc_env['QM_reps']):
                X2 = makeX_adiab_Qlk(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm, irep)
            else:
                X2 = makeX_adiab_ehren(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm)
#            X2 = makeX_adiab_Qlk(ctmqc_env, H_tm, NACV_tm, v_tm, E_tm, irep) 

            # RK4
            coeff = __RK4(ctmqc_env['C'][irep], X1, X12, X2,
                          ctmqc_env)
            ctmqc_env['C'][irep] = coeff

            X1 = X2[:]  # Update X_{1}


def makeX_adiab_Qlk(ctmqc_env, H, NACV, vel, E, irep):
    """
    Will make the adiabatic X matrix with Qlk
    """
    nstates = ctmqc_env['nstate']
    X = np.zeros((nstates, nstates), dtype=complex)
    Xqm = np.zeros((nstates, nstates), dtype=complex)

    # Ehrenfest Part
    X = (-1j * np.identity(2) * E) - (NACV * vel)

    # QM Part
    # Calculate QM, adMom and adPops
    C = ctmqc_env['C'][irep]  # coeff

    adPops = np.conjugate(C) * C
    Qlk = ctmqc_env['Qlk'][irep]
    f = ctmqc_env['adMom'][irep]

    # ammend X
    for l in range(nstates):
        for k in range(nstates):
            Xqm[l, l] += Qlk[l, k] * (f[k] - f[l]) * adPops[k]

    return np.array(X - Xqm)


def __RK4_2(ctmqc_env, coeff, X1, X12, X2):
    """
    Will carry out the RK4 algorithm to propagate the coefficients
    """
    dTe = ctmqc_env['dt'] / float(ctmqc_env['elec_steps'])
    coeff = np.array(coeff)

    K1 = np.array(dTe * np.matmul(X1, coeff))[0]
    K2 = np.array(dTe * np.matmul(X12, coeff + K1/2.))[0]
    K3 = np.array(dTe * np.matmul(X12, coeff + K2/2.))[0]
    K4 = np.array(dTe * np.matmul(X2, coeff + K3))[0]

    Ktot = (1./6.) * (K1 + (2.*K2) + (2.*K3) + K4)

    coeff = coeff + Ktot

    return coeff


def __RK4(coeff, X1, X12, X2, ctmqc_env):
    """
    Will carry out the RK4 algorithm to propagate the coefficients
    """
    dTe = ctmqc_env['dt'] / float(ctmqc_env['elec_steps'])
    coeff = np.array(coeff)

    K1 = np.array(dTe * np.dot(np.array(X1), coeff))
    K2 = np.array(dTe * np.dot(np.array(X12), coeff + K1/2.))
    K3 = np.array(dTe * np.dot(np.array(X12), coeff + K2/2.))
    K4 = np.array(dTe * np.dot(np.array(X2), coeff + K3))
    Ktot = (1./6.) * (K1 + (2.*K2) + (2.*K3) + K4)
    coeff = coeff + Ktot

    return coeff

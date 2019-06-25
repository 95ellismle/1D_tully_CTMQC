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


def renormalise_all_coeffs(coeff):
    """
    Will renormalise all the coefficients for replica I, atom v.
    """
    nrep, nstate = np.shape(coeff)
    norms = np.linalg.norm(coeff, axis=2)
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


def do_diab_prop_ehren(ctmqc_env):
    """
    Will propagate the coefficients in the diabatic basis (without the
    diabatic NACE)
    """
    for irep in range(ctmqc_env['nrep']):
        dx_E = (ctmqc_env['pos'][irep] - ctmqc_env['pos_tm'][irep])
        dx_E /= float(ctmqc_env['elec_steps'])
    
        X1 = makeX_diab(ctmqc_env, ctmqc_env['pos_tm'], irep)
        for Estep in range(ctmqc_env['elec_steps']):
            pos2 = ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
            pos3 = ctmqc_env['pos_tm'] + (Estep+1) * dx_E
    
            X12 = makeX_diab(ctmqc_env, pos2, irep)
            X2 = makeX_diab(ctmqc_env, pos3, irep)
    
            coeff = __RK4(ctmqc_env['u'][irep], X1, X12, X2, ctmqc_env)
            ctmqc_env['u'][irep] = coeff
    
            X1 = X2[:]


def makeX_diab(ctmqc_env, pos, irep):
    """
    Will make the diabatic X matrix
    """
    H = ctmqc_env['Hfunc'](pos[irep])

    return -1j * H


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

            coeff = __RK4(ctmqc_env['C'][irep], X1, X12, X2,
                          ctmqc_env)
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
    E, U = Ham.getEigProps(H, ctmqc_env)
    NACV = Ham.calcNACV(irep, ctmqc_env)
    vel = ctmqc_env['vel'][irep]
    
    # First part
    for l in range(nstates):
        X[l, l] = E[l]

    # Second part
    X += -1j * NACV * vel

    return -1j * np.array(X)


def do_adiab_prop_QM(ctmqc_env, allTimes):
    """
    Will actually carry out the propagation of the coefficients
    """
    makeX_time = 0.0
    linInterp = 0.0
    RK4_time = 0.0
    for irep in range(ctmqc_env['nrep']):
        t1 = time.time()
        dx_E = (ctmqc_env['pos'][irep]
                - ctmqc_env['pos_tm'][irep])
        dx_E /= float(ctmqc_env['elec_steps'])
        dv_E = (ctmqc_env['vel'][irep]
                - ctmqc_env['vel_tm'][irep])
        dv_E /= float(ctmqc_env['elec_steps'])
        dQlk_E = (ctmqc_env['Qlk'][irep]
                  - ctmqc_env['Qlk_tm'][irep])
        dQlk_E /= float(ctmqc_env['elec_steps'])
        df_E = (ctmqc_env['adMom'][irep]
                - ctmqc_env['adMom_tm'][irep])
        df_E /= float(ctmqc_env['elec_steps'])

        # Make X_{1}
        ctmqc_env['vel'][irep] = ctmqc_env['vel_tm'][irep]
        ctmqc_env['Qlk'][irep] = ctmqc_env['Qlk_tm'][irep]
        ctmqc_env['adMom'][irep] = ctmqc_env['adMom_tm'][irep]
        linInterp += time.time() - t1

        t1 = time.time()
        X1 = makeX_adiab_Qlk(ctmqc_env, ctmqc_env['pos_tm'], irep)
        makeX_time += time.time() - t1

        for Estep in range(ctmqc_env['elec_steps']):
            t1 = time.time()
            # Linear Interpolationprint(check)
            ctmqc_env['Qlk'][irep] = ctmqc_env['Qlk_tm'][irep]\
                + (Estep + 0.5) * dQlk_E
            ctmqc_env['adMom'][irep] = ctmqc_env['adMom_tm'][irep]\
                + (Estep + 0.5) * df_E
            pos2 = ctmqc_env['pos_tm'][irep] + (Estep + 0.5) * dx_E
            ctmqc_env['vel'][irep] = ctmqc_env['vel_tm'][irep]\
                + (Estep + 0.5) * dv_E
            linInterp += time.time() - t1

            # Make X12
            t1 = time.time()
            X12 = makeX_adiab_Qlk(ctmqc_env, pos2, irep)
            makeX_time += time.time() - t1

            # Linear Interpolation
            t1 = time.time()
            pos3 = ctmqc_env['pos_tm'][irep] + (Estep+1) * dx_E
            ctmqc_env['Qlk'][irep] = ctmqc_env['Qlk_tm'][irep]\
                + (Estep + 1.0) * dQlk_E
            ctmqc_env['adMom'][irep] = ctmqc_env['adMom_tm'][irep]\
                + (Estep + 1.0) * df_E
            ctmqc_env['vel'][irep] = ctmqc_env['vel_tm'][irep]\
                + (Estep + 1) * dv_E
            linInterp += time.time() - t1

            # Make X2
            t1 = time.time()
            X2 = makeX_adiab_Qlk(ctmqc_env, pos3, irep)
            makeX_time += time.time() - t1

            t1 = time.time()
            # RK4
            coeff = __RK4(ctmqc_env['C'][irep], X1, X12, X2,
                          ctmqc_env)
            ctmqc_env['C'][irep] = coeff
            RK4_time += time.time() - t1

            t1 = time.time()
            X1 = X2[:]  # Update X_{1}
            linInterp += time.time() - t1

    allTimes['wf_prop']['prop']['makeX'].append(makeX_time)
    allTimes['wf_prop']['prop']['RK4'].append(RK4_time)
    allTimes['wf_prop']['prop']['lin. interp'].append(RK4_time)


def makeX_adiab_Qlk(ctmqc_env, pos, irep):
    """
    Will make the adiabatic X matrix with Qlk
    """
    nstates = ctmqc_env['nstate']
    X = np.zeros((nstates, nstates), dtype=complex)
    Xqm = np.zeros((nstates, nstates), dtype=complex)
    pos = ctmqc_env['pos']

    H = ctmqc_env['Hfunc'](pos[irep])
    E, U = Ham.getEigProps(H, ctmqc_env)
    NACV = Ham.calcNACV(irep, ctmqc_env)
    vel = ctmqc_env['vel'][irep]

    # Ehrenfest Part
    X = (-1j * np.identity(2) * E) - (NACV * vel)

    # QM Part
    # Calculate QM, adMom and adPops
    C = ctmqc_env['C'][irep]  # coeff

    adPops = calc_ad_pops(C, ctmqc_env)
    Qlk = ctmqc_env['Qlk'][irep]
    f = ctmqc_env['adMom'][irep]

    # ammend X
    for l in range(nstates):
        for k in range(nstates):
            Xqm[l, l] += Qlk[l, k] * (f[k] - f[l]) * adPops[k]

    return np.array(X - Xqm)


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

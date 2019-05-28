#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:10 2019

@author: mellis
"""
import numpy as np

import hamiltonian as Ham
import QM_utils as qUt


def trans_diab_to_adiab(H, u, ctmqc_env):
    """
    Will transform the diabatic coefficients to adiabatic ones
    """
    if len(u) != 2 and len(np.shape(u)) != 1:
        raise SystemExit("Incorrect Shape for diab coeff in trans func")

    U = Ham.getEigProps(H, ctmqc_env)[1]

    return np.dot(np.array(U), np.array(u))


def trans_adiab_to_diab(H, C, ctmqc_env):
    """
    Will transform the adiabatic coefficients to diabatic ones
    """
    if len(C) != 2 and len(np.shape(C)) != 1:
        raise SystemExit("Incorrect Shape for diab coeff in trans func")

    U = Ham.getEigProps(H, ctmqc_env)[1]

    return np.dot(np.array(U.T), np.array(C))


def renormalise_all_coeffs(coeff):
    """
    Will renormalise all the coefficients for replica I, atom v.
    """
    return coeff
#    norms = np.linalg.norm(coeff, axis=2)
#    print(coeff.shape)
#    coeff = coeff[:, :] / norms
#    print(coeff.shape)
#    raise SystemExit("BREAK")
#    return coeff


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


def do_diab_prop_ehren(ctmqc_env, irep, iatom):
    """
    Will propagate the coefficients in the diabatic basis (without the
    diabatic NACE)
    """
    dx_E = (ctmqc_env['pos'] - ctmqc_env['pos_tm'])
    dx_E /= float(ctmqc_env['elec_steps'])

    X1 = makeX_diab(ctmqc_env, ctmqc_env['pos_tm'], irep, iatom)
    for Estep in range(ctmqc_env['elec_steps']):
        pos2 = ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
        pos3 = ctmqc_env['pos_tm'] + (Estep+1) * dx_E

        X12 = makeX_diab(ctmqc_env, pos2, irep, iatom)
        X2 = makeX_diab(ctmqc_env, pos3, irep, iatom)

        coeff = __RK4(ctmqc_env['u'][irep, iatom], X1, X12, X2, ctmqc_env)
        ctmqc_env['u'][irep, iatom] = coeff

        X1 = X2[:]


def makeX_diab(ctmqc_env, pos, irep, iatom):
    """
    Will make the diabatic X matrix
    """
    H = ctmqc_env['Hfunc'](pos[irep, iatom])

    return -1j * H


def do_diab_prop_QM(ctmqc_env, irep, iatom):
    """
    Will propagate the coefficients in the diabatic basis (without the
    diabatic NACE)
    """
    dx_E = (ctmqc_env['pos'] - ctmqc_env['pos_tm'])
    dx_E /= float(ctmqc_env['elec_steps'])

    X1 = makeX_diab_QM(ctmqc_env, ctmqc_env['pos_tm'], irep, iatom)
    for Estep in range(ctmqc_env['elec_steps']):
        pos2 = ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
        pos3 = ctmqc_env['pos_tm'] + (Estep + 1.0) * dx_E

        X12 = makeX_diab_QM(ctmqc_env, pos2, irep, iatom)
        X2 = makeX_diab_QM(ctmqc_env, pos3, irep, iatom)

        coeff = __RK4(ctmqc_env['u'][irep, iatom], X1, X12, X2, ctmqc_env)
        ctmqc_env['u'][irep, iatom] = coeff

        X1 = X2[:]


def makeX_diab_QM(ctmqc_env, pos, irep, iatom):
    """
    Will make the diabatic X matrix
    """
    nstate = ctmqc_env['nstate']
    H = ctmqc_env['Hfunc'](pos[irep, iatom])
    U = Ham.getEigProps(H, ctmqc_env)[1]
    X = -1j * H
    K = np.zeros((nstate, nstate))

    C = trans_diab_to_adiab(H,
                            ctmqc_env['u'][irep, iatom],
                            ctmqc_env)
    ctmqc_env['C'][irep, iatom] = C
    adPops = calc_ad_pops(C, ctmqc_env)

    ctmqc_env['pos'] = pos
    QM = qUt.calc_QM_analytic(ctmqc_env, irep, iatom)
    adMom = qUt.calc_ad_mom(ctmqc_env, irep, iatom)
    for l in range(nstate):
        tmp = 0.0
        for k in range(nstate):
            Ck = adPops[k]
            fk = adMom[k]
            tmp += Ck*fk

        K[l, l] = QM * adMom[l] * tmp

    Xctmqc = np.dot(U.T, np.dot(K, U))

    return X - Xctmqc


def do_adiab_prop_ehren(ctmqc_env, irep, iatom):
    """
    Will actually carry out the propagation of the coefficients
    """
    dx_E = (ctmqc_env['pos'] - ctmqc_env['pos_tm'])
    dx_E /= float(ctmqc_env['elec_steps'])

    ctmqc_env['pos'] = ctmqc_env['pos_tm']
    X1 = makeX_adiab(ctmqc_env, irep, iatom)
    for Estep in range(ctmqc_env['elec_steps']):
        ctmqc_env['pos'] = ctmqc_env['pos_tm'] \
                                + (Estep + 0.5) * dx_E
        X12 = makeX_adiab(ctmqc_env, irep, iatom)

        ctmqc_env['pos'] = ctmqc_env['pos_tm'] \
            + (Estep+1) * dx_E
        X2 = makeX_adiab(ctmqc_env, irep, iatom)

        coeff = __RK4(ctmqc_env['C'][irep, iatom], X1, X12, X2, ctmqc_env)
        ctmqc_env['C'][irep, iatom] = coeff

        X1 = X2[:]


def makeX_adiab(ctmqc_env, irep, iatom):
    """
    Will make the adiabatic X matrix
    """
    nstates = ctmqc_env['nstate']
    X = np.zeros((nstates, nstates), dtype=complex)
    pos = ctmqc_env['pos']

    H = ctmqc_env['Hfunc'](pos[irep, iatom])
    E, U = Ham.getEigProps(H, ctmqc_env)
    NACV = Ham.calcNACV(irep, iatom, ctmqc_env)
    vel = ctmqc_env['vel'][irep, iatom]
    # First part
    for l in range(nstates):
        X[l, l] = E[l]

    # Second part
    X += -1j * NACV * vel

    return -1j * np.array(X)


def do_adiab_prop_QM(ctmqc_env, irep, iatom):
    """
    Will actually carry out the propagation of the coefficients
    """
    dx_E = (ctmqc_env['pos'][irep, iatom] - ctmqc_env['pos_tm'][irep, iatom])
    dx_E /= float(ctmqc_env['elec_steps'])
    dv_E = (ctmqc_env['vel'][irep, iatom] - ctmqc_env['vel_tm'][irep, iatom])
    dv_E /= float(ctmqc_env['elec_steps'])
    dQM_E = (ctmqc_env['QM'][irep, iatom] - ctmqc_env['QM_tm'][irep, iatom])
    dQM_E /= float(ctmqc_env['elec_steps'])
    df_E = (ctmqc_env['adMom'][irep, iatom] - ctmqc_env['adMom_tm'][irep,
                                                                    iatom])
    df_E /= float(ctmqc_env['elec_steps'])

    # Make X_{1}
    ctmqc_env['vel'][irep, iatom] = ctmqc_env['vel_tm'][irep, iatom]
    ctmqc_env['QM'][irep, iatom] = ctmqc_env['QM_tm'][irep, iatom]
    ctmqc_env['adMom'][irep, iatom] = ctmqc_env['adMom_tm'][irep, iatom]

    X1 = makeX_adiab_QM(ctmqc_env, ctmqc_env['pos_tm'], irep, iatom)
    for Estep in range(ctmqc_env['elec_steps']):
        # Linear Interpolationprint(check)
        ctmqc_env['QM'][irep, iatom] = ctmqc_env['QM_tm'][irep, iatom] \
            + (Estep + 0.5) * dQM_E
        ctmqc_env['adMom'][irep, iatom] = ctmqc_env['adMom_tm'][irep, iatom] \
            + (Estep + 0.5) * df_E
        pos2 = ctmqc_env['pos_tm'][irep, iatom] + (Estep + 0.5) * dx_E
        ctmqc_env['vel'][irep, iatom] = ctmqc_env['vel_tm'][irep, iatom] \
            + (Estep + 0.5) * dv_E

        # Make X12
        X12 = makeX_adiab_QM(ctmqc_env, pos2, irep, iatom)

        # Linear Interpolation
        pos3 = ctmqc_env['pos_tm'][irep, iatom] + (Estep+1) * dx_E
        ctmqc_env['QM'][irep, iatom] = ctmqc_env['QM_tm'][irep, iatom] \
            + (Estep + 1.0) * dQM_E
        ctmqc_env['adMom'][irep, iatom] = ctmqc_env['adMom_tm'][irep, iatom] \
            + (Estep + 1.0) * df_E
        ctmqc_env['vel'][irep, iatom] = ctmqc_env['vel_tm'][irep, iatom] \
            + (Estep + 1) * dv_E

        # Make X2
        X2 = makeX_adiab_QM(ctmqc_env, pos3, irep, iatom)

        # RK4
        coeff = __RK4(ctmqc_env['C'][irep, iatom], X1, X12, X2, ctmqc_env)
        ctmqc_env['C'][irep, iatom] = coeff

        X1 = X2[:]  # Update X_{1}


def makeX_adiab_QM(ctmqc_env, pos, irep, iatom):
    """
    Will make the adiabatic X matrix with QM
    """
    nstates = ctmqc_env['nstate']
    X = np.zeros((nstates, nstates), dtype=complex)
    Xeh = np.zeros((nstates, nstates), dtype=complex)
    Xqm = np.zeros((nstates, nstates), dtype=complex)
    pos = ctmqc_env['pos']

    H = ctmqc_env['Hfunc'](pos[irep, iatom])
    E, U = Ham.getEigProps(H, ctmqc_env)
    NACV = Ham.calcNACV(irep, iatom, ctmqc_env)
    vel = ctmqc_env['vel'][irep, iatom]

    # Ehrenfest Part
    for l in range(nstates):
        X[l, l] = -1j * E[l]

    Xeh -= NACV * vel

    # QM Part
    # Calculate QM, adMom and adPops
    C = ctmqc_env['C'][irep, iatom]  # coeff

    adPops = calc_ad_pops(C, ctmqc_env)
    QM = ctmqc_env['QM'][irep, iatom]
    adMom = ctmqc_env['adMom'][irep, iatom]

    # ammend X
    tmp = np.sum(adPops * adMom)  # The sum over k part
    for l in range(nstates):
        Xqm[l, l] += QM * (tmp - adMom[l])

    check = np.sum(adPops * np.diagonal(Xqm))
    if check > 1e-5:
        print(check)

    X = np.array(Xeh + Xqm)
    return X


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

from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:10 2019

@author: mellis
"""
import numpy as np


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


def do_diab_prop(ctmqc_env):
    """
    Will propagate the coefficients in the diabatic basis (without the
    diabatic NACE)  
    
    N.B. Is just Ehrenfest at the moment
    """
    for irep in range(ctmqc_env['nrep']):
        H = np.matrix(ctmqc_env['H_tm'][irep])
        dH_E = get_diffVal(ctmqc_env['H'][irep], H, ctmqc_env)
        
        U = np.matrix(ctmqc_env['U_tm'][irep])
        dU_E = get_diffVal(ctmqc_env['U'][irep], U, ctmqc_env)

        QM = ctmqc_env['Qlk_tm'][irep]
        dQM_E = get_diffVal(ctmqc_env['Qlk'][irep], QM, ctmqc_env)

        f = ctmqc_env['adMom_tm'][irep]
        df_E = get_diffVal(ctmqc_env['adMom'][irep], f, ctmqc_env)
        
        adPops = np.conjugate(ctmqc_env['C'][irep]) * ctmqc_env['C'][irep]
        doQM = abs(QM[0, 1]) > 1e-10
        
        X1 = makeX_diab_ehren(H)
        if doQM: X1 -= makeX_diab_QM(QM, f, U, adPops)
        for Estep in range(ctmqc_env['elec_steps']):
            H += 0.5 * dH_E
            U += 0.5 * dU_E
            QM += 0.5 * dQM_E
            f += 0.5 * df_E
            X12 = makeX_diab_ehren(H)
            if doQM: X12 -= makeX_diab_QM(QM, f, U, adPops)

            X2 = makeX_diab_ehren(H)
            if doQM: X2 -= makeX_diab_QM(QM, f, U, adPops)

            H += 0.5 * dH_E
            U += 0.5 * dU_E
            QM += 0.5 * dQM_E
            f += 0.5 * df_E
            ctmqc_env['u'][irep] = __RK4(ctmqc_env['u'][irep], X1.A, X12.A,
                                         X2.A, ctmqc_env)

            C = np.matmul(np.array(U.T), ctmqc_env['u'][irep])
            adPops = np.conjugate(C) * C

            X1 = X2[:]
        
        lin_interp_check(ctmqc_env['H'][irep], H, "Hamiltonian")
        lin_interp_check(ctmqc_env['adMom'][irep], f, "Adiabatic Momentum")
        lin_interp_check(ctmqc_env['Qlk'][irep], QM, "Quantum Momentum")
#        lin_interp_check(ctmqc_env['U'][irep], U, "Quantum Momentum")


def makeX_diab_QM(Qlk, f, U, adPops):
    """
    Will make the diabatic X matrix for the full quantum momentum propagation.
    
    N.B. only give Ehrenfest atm
    """
    nstates = len(adPops)
    Xqm = np.zeros((nstates, nstates), dtype=complex)

    # make X
    for l in range(nstates):

        for k in range(nstates):
            Xqm[l, l] += Qlk[l, k] * (f[k] - f[l]) * adPops[k]
    
    # Check the Xqm term (using norm conservation)
    if np.sum(Xqm * adPops) > 1e-10:
        print("\n\nQM: ", Qlk, "\n")
        print("f: ", f, "\n")
        print("adPops: ", adPops, "\n")
        print("Xqm: ", Xqm, "\n")
        print("Xqm * adPops: ", Xqm * adPops, "\n")
        raise SystemExit("ERROR: MakeX, sum Xqm != 0")
    
    Xqm = np.matmul(U, np.matmul(Xqm, U.T))
    
    return Xqm


def lin_interp_check(oldVal, interpVal, name):
    """
    Will check if the linear interpolation went well for the named variable
    """
    if np.max(oldVal) > 1e-13:
        if abs(np.max(oldVal - interpVal)/np.max(oldVal)) > 1e-5:
            print("\n\nOld Value = ", oldVal)
            print("\n\nInterpolated Value = ", interpVal)
            print("\n\nDifference = ", oldVal - interpVal)
            print("Quantity = %s" % name)
            raise SystemExit("Something went wrong with the linear " +
                                 "interpolation of the %s" % name)
    


def makeX_adiab_ehren(NACV, vel, E):
    """
    Will make the adiabatic X matrix
    """
    X = (-1j * np.identity(2) * E) - (NACV * vel)
    
    return X


def do_adiab_prop(ctmqc_env):
    """
    Will actually carry out the propagation of the coefficients
    """
    for irep in range(ctmqc_env['nrep']):
        v = ctmqc_env['vel_tm'][irep]
        dv_E = get_diffVal(ctmqc_env['vel'][irep], v, ctmqc_env)

        E = ctmqc_env['E_tm'][irep]
        dE_E = get_diffVal(ctmqc_env['E'][irep], E, ctmqc_env)

        NACV = ctmqc_env['NACV_tm'][irep]
        dNACV_E = get_diffVal(ctmqc_env['NACV'][irep], NACV, ctmqc_env)

        QM = ctmqc_env['Qlk_tm'][irep]
        dQM_E = get_diffVal(ctmqc_env['Qlk'][irep], QM, ctmqc_env)

        f = ctmqc_env['adMom_tm'][irep]
        df_E = get_diffVal(ctmqc_env['adMom'][irep], f, ctmqc_env)
        
#        print("VARS")
#        print("----")
#        print("iter = ", ctmqc_env['iter'])
#        print("----")
#        print("Vel = ", v)
#        print("dv = ", dv_E)
#        print("----")
#        print("E = ", E)
#        print("dE = ", dE_E)
#        print("----")
#        print("dlk = ", NACV)
#        print("d_dlk = ", dNACV_E)
#        print("----")
#        print("H = ", ctmqc_env['H'][irep])
#        print("----")
#        print("|C|^2 = ", np.conjugate(ctmqc_env['C'][irep]) * ctmqc_env['C'][irep])
#        print("----")
#        print("Qlk_t = ", QM[0, 1])
#        print("Qlk_tp = ", ctmqc_env['Qlk'][irep][0, 1])
#        print("dQlk = ", dQM_E)
#        print("----")
#        print("f_t = ", f)
#        print("f_tp = ", ctmqc_env['adMom'][irep])
#        print("df = ", df_E)
#        print("END VARS")
#        raise SystemExit("BREAK")
        
        adPops = np.conjugate(ctmqc_env['C'][irep]) * ctmqc_env['C'][irep]
        doQM = abs(QM[0, 1]) > 1e-10
        

        X1 = makeX_adiab_ehren(NACV, v, E)
        if doQM: X1 -= makeX_adiab_Qlk(QM, f, adPops)
#        print("X1: ", X1)
        for Estep in range(ctmqc_env['elec_steps']):
            E += 0.5 * dE_E
            NACV += 0.5 * dNACV_E
            v += 0.5 * dv_E
            QM += 0.5 * dQM_E
            f += 0.5 * df_E
            X12 = makeX_adiab_ehren(NACV, v, E)
            if doQM:
                adPops = np.conjugate(ctmqc_env['C'][irep]) * ctmqc_env['C'][irep]
                X12 -= makeX_adiab_Qlk(QM, f, adPops)
#            print("X12: ", X12)

            E = E + 0.5 * dE_E
            NACV = NACV + 0.5 * dNACV_E
            v = v + 0.5 * dv_E
            QM += 0.5 * dQM_E
            f += 0.5 * df_E
            X2 = makeX_adiab_ehren(NACV, v, E)
            if doQM: X2 -= makeX_adiab_Qlk(QM, f, adPops)
#            print("X2: ", X2)

            coeff = __RK4(ctmqc_env['C'][irep], X1, X12, X2,
                          ctmqc_env)
            ctmqc_env['C'][irep] = coeff

            X1 = X2[:]
        
#        lin_interp_check(ctmqc_env['H'][irep], H, "Hamiltonian")
        lin_interp_check(ctmqc_env['NACV'][irep], NACV, "NACV")
        lin_interp_check(ctmqc_env['E'][irep], E, "Energy")
        lin_interp_check(ctmqc_env['vel'][irep], v, "Velocity")
        lin_interp_check(ctmqc_env['adMom'][irep], f, "Adiabatic Momentum")
        lin_interp_check(ctmqc_env['Qlk'][irep], QM, "Quantum Momentum")


def makeX_adiab_Qlk(Qlk, f, adPops):
    """
    Will make the adiabatic X matrix with Qlk
    """
    nstates = len(adPops)
    Xqm = np.zeros((nstates, nstates), dtype=complex)

    # make X
    for l in range(nstates):

        for k in range(nstates):
            Xqm[l, l] += Qlk[l, k] * (f[k] - f[l]) * adPops[k]
    
    # Check the Xqm term (using norm conservation)
    if np.sum(Xqm * adPops) > 1e-10:
        print("\n\nQM: ", Qlk, "\n")
        print("f: ", f, "\n")
        print("adPops: ", adPops, "\n")
        print("Xqm: ", Xqm, "\n")
        print("Xqm * adPops: ", Xqm * adPops, "\n")
        raise SystemExit("ERROR: MakeX, sum Xqm != 0")
    
    return Xqm


def __RK4(coeff, X1, X12, X2, ctmqc_env):
    """
    Will carry out the RK4 algorithm to propagate the coefficients
    """
    dTe = ctmqc_env['dt'] / float(ctmqc_env['elec_steps'])
    coeff = np.array(coeff)

    K1 = np.array(dTe * np.matmul(X1, coeff))
    K2 = np.array(dTe * np.matmul(X12, coeff + K1/2.))
    K3 = np.array(dTe * np.matmul(X12, coeff + K2/2.))
    K4 = np.array(dTe * np.matmul(X2, coeff + K3))

    Ktot = (1./6.) * (K1 + (2.*K2) + (2.*K3) + K4)
    coeff = coeff + Ktot

    return coeff


'''






def do_diab_prop_ehren(ctmqc_env):
    """
    Will propagate the coefficients in the diabatic basis (without the
    diabatic NACE)
    """
    for irep in range(ctmqc_env['nrep']):
        H = np.matrix(ctmqc_env['H_tm'][irep])
        dH_E = get_diffVal(ctmqc_env['H'][irep], H, ctmqc_env)

        X1 = makeX_diab_ehren(H)
        for Estep in range(ctmqc_env['elec_steps']):
            H += 0.5 * dH_E
            X12 = makeX_diab_ehren(H)

            X2 = makeX_diab_ehren(H)

            H += 0.5 * dH_E
            ctmqc_env['u'][irep] = __RK4(ctmqc_env['u'][irep], X1.A, X12.A,
                                         X2.A, ctmqc_env)

            X1 = X2[:]
    

def do_adiab_prop_ehren(ctmqc_env):
    """
    Will actually carry out the propagation of the coefficients
    """
    for irep in range(ctmqc_env['nrep']):
        v = ctmqc_env['vel_tm'][irep]
        dv_E = get_diffVal(ctmqc_env['vel'][irep], v, ctmqc_env)

        E = ctmqc_env['E_tm'][irep]
        dE_E = get_diffVal(ctmqc_env['E'][irep], E, ctmqc_env)

        NACV = ctmqc_env['NACV_tm'][irep]
        dNACV_E = get_diffVal(ctmqc_env['NACV'][irep], NACV, ctmqc_env)

        X1 = makeX_adiab_ehren(NACV, v, E)
        for Estep in range(ctmqc_env['elec_steps']):
            E += 0.5 * dE_E
            NACV += 0.5 * dNACV_E
            v += 0.5 * dv_E
            X12 = makeX_adiab_ehren(NACV, v, E)

            E = E + 0.5 * dE_E
            NACV = NACV + 0.5 * dNACV_E
            v = v + 0.5 * dv_E
            X2 = makeX_adiab_ehren(NACV, v, E)

            coeff = __RK4(ctmqc_env['C'][irep], X1, X12, X2,
                          ctmqc_env)
            ctmqc_env['C'][irep] = coeff

            X1 = X2[:]
        
        lin_interp_check(ctmqc_env['NACV'][irep], NACV, "NACV")
        lin_interp_check(ctmqc_env['E'][irep], E, "Energy")
        lin_interp_check(ctmqc_env['vel'][irep], v, "Velocity")
'''

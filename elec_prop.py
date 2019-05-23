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


class elecProp(object):
    """
    Will handle the electronic propagation (make the X matrix and feed it into
    RK4)
    """
    def __init__(self, ctmqc_env):
        self.ctmqc_env = ctmqc_env
        self.dTe = self.ctmqc_env['dt'] / float(self.ctmqc_env['elec_steps'])

    def do_diab_prop_QM(self, irep, iatom):
        """
        Will propagate the coefficients in the diabatic basis (without the
        diabatic NACE)
        """
        dx_E = (self.ctmqc_env['pos'] - self.ctmqc_env['pos_tm'])
        dx_E /= float(self.ctmqc_env['elec_steps'])

        self.X1 = self.makeX_diab_QM(self.ctmqc_env['pos_tm'], irep, iatom)
        for Estep in range(self.ctmqc_env['elec_steps']):
            pos2 = self.ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
            pos3 = self.ctmqc_env['pos_tm'] + (Estep+1) * dx_E

            self.X12 = self.makeX_diab_QM(pos2, irep, iatom)
            self.X2 = self.makeX_diab_QM(pos3, irep, iatom)

            coeff = self.__RK4(self.ctmqc_env['u'][irep, iatom])
            self.ctmqc_env['u'][irep, iatom] = coeff

            self.X1 = self.X2[:]

    def do_adiab_prop_QM(self, irep, iatom):
        """
        Will actually carry out the propagation of the coefficients
        """
        dx_E = (self.ctmqc_env['pos'] - self.ctmqc_env['pos_tm'])
        dx_E /= float(self.ctmqc_env['elec_steps'])
        dv_E = (self.ctmqc_env['vel'] - self.ctmqc_env['vel_tm'])
        dv_E /= float(self.ctmqc_env['elec_steps'])

        # Make X_{1}
        self.ctmqc_env['vel'] = self.ctmqc_env['vel_tm']
        self.X1 = self.makeX_adiab_QM(self.ctmqc_env['pos_tm'], irep, iatom)
        for Estep in range(self.ctmqc_env['elec_steps']):
            # Make X_{1/2}
            pos2 = self.ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
            self.ctmqc_env['vel'] = self.ctmqc_env['vel_tm'] \
                + (Estep + 0.5) * dv_E
            self.X12 = self.makeX_adiab_QM(pos2, irep, iatom)

            # Make X_2
            pos3 = self.ctmqc_env['pos_tm'] + (Estep+1) * dx_E
            self.ctmqc_env['vel'] = self.ctmqc_env['vel_tm'] \
                + (Estep + 1) * dv_E
            self.X2 = self.makeX_adiab_QM(pos3, irep, iatom)

            coeff = self.__RK4(self.ctmqc_env['C'][irep, iatom])
            self.ctmqc_env['C'][irep, iatom] = coeff

            self.X1 = self.X2[:]  # Update X_{1}

    def do_diab_prop_ehren(self, irep, iatom):
        """
        Will propagate the coefficients in the diabatic basis (without the
        diabatic NACE)
        """
        dx_E = (self.ctmqc_env['pos'] - self.ctmqc_env['pos_tm'])
        dx_E /= float(self.ctmqc_env['elec_steps'])

        self.X1 = self.makeX_diab(self.ctmqc_env['pos_tm'], irep, iatom)
        for Estep in range(self.ctmqc_env['elec_steps']):
            pos2 = self.ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
            pos3 = self.ctmqc_env['pos_tm'] + (Estep+1) * dx_E

            self.X12 = self.makeX_diab(pos2, irep, iatom)
            self.X2 = self.makeX_diab(pos3, irep, iatom)

            coeff = self.__RK4(self.ctmqc_env['u'][irep, iatom])
            self.ctmqc_env['u'][irep, iatom] = coeff

            self.X1 = self.X2[:]

    def do_adiab_prop_ehren(self, irep, iatom):
        """
        Will actually carry out the propagation of the coefficients
        """
        dx_E = (self.ctmqc_env['pos'] - self.ctmqc_env['pos_tm'])
        dx_E /= float(self.ctmqc_env['elec_steps'])

        self.ctmqc_env['pos'] = self.ctmqc_env['pos_tm']
        self.X1 = self.makeX_adiab(irep, iatom)
        for Estep in range(self.ctmqc_env['elec_steps']):
            self.ctmqc_env['pos'] = self.ctmqc_env['pos_tm'] \
                                    + (Estep + 0.5) * dx_E
            self.X12 = self.makeX_adiab(irep, iatom)

            self.ctmqc_env['pos'] = self.ctmqc_env['pos_tm'] \
                                    + (Estep+1) * dx_E
            self.X2 = self.makeX_adiab(irep, iatom)

            coeff = self.__RK4(self.ctmqc_env['C'][irep, iatom])
            self.ctmqc_env['C'][irep, iatom] = coeff

            self.X1 = self.X2[:]

    def makeX_diab(self, pos, irep, iatom):
        """
        Will make the diabatic X matrix
        """
        H = self.ctmqc_env['Hfunc'](pos[irep, iatom])

        return -1j * H

    def makeX_diab_QM(self, pos, irep, iatom):
        """
        Will make the diabatic X matrix
        """
        nstate = self.ctmqc_env['nstate']
        H = self.ctmqc_env['Hfunc'](pos[irep, iatom])
        U = Ham.getEigProps(H, self.ctmqc_env)[1]
        X = -1j * H
        K = np.zeros((nstate, nstate))

        C = trans_diab_to_adiab(H,
                                self.ctmqc_env['u'][irep, iatom],
                                self.ctmqc_env)
        self.ctmqc_env['C'][irep, iatom] = C
        self.ctmqc_env['adPops'][irep, iatom] = calc_ad_pops(C, self.ctmqc_env)

        QM = qUt.calc_QM(self.ctmqc_env['adPops'][irep, iatom],
                         self.ctmqc_env,
                         irep, iatom)
        self.ctmqc_env['pos'] = pos
        adMom = qUt.calc_ad_mom(self.ctmqc_env, irep, iatom)

        for l in range(nstate):
            tmp = 0.0
            for k in range(nstate):
                Ck = self.ctmqc_env['adPops'][irep, k]
                fk = adMom[k]
                tmp += Ck*fk

            K[l, l] = QM * adMom[l] * tmp

        Xctmqc = np.dot(U.T, np.dot(K, U))

        return X - Xctmqc

    def makeX_adiab(self, irep, iatom):
        """
        Will make the adiabatic X matrix
        """
        nstates = self.ctmqc_env['nstate']
        X = np.zeros((nstates, nstates), dtype=complex)
        pos = self.ctmqc_env['pos']

        H = self.ctmqc_env['Hfunc'](pos[irep, iatom])
        E, U = Ham.getEigProps(H, self.ctmqc_env)
        NACV = Ham.calcNACV(irep, iatom, self.ctmqc_env)
        vel = self.ctmqc_env['vel'][irep, iatom]
        # First part
        for l in range(nstates):
            X[l, l] = E[l]

        # Second part
        X += -1j * NACV * vel

        return -1j * np.array(X)

    def makeX_adiab_QM(self, pos, irep, iatom):
        """
        Will make the adiabatic X matrix with QM
        """
        nstates = self.ctmqc_env['nstate']
        X = np.zeros((nstates, nstates), dtype=complex)
        pos = self.ctmqc_env['pos']

        H = self.ctmqc_env['Hfunc'](pos[irep, iatom])
        E, U = Ham.getEigProps(H, self.ctmqc_env)
        NACV = Ham.calcNACV(irep, iatom, self.ctmqc_env)
        vel = self.ctmqc_env['vel'][irep, iatom]
        # First part
        for l in range(nstates):
            X[l, l] = E[l]

        # Second part
        X += -1j * NACV * vel
        X = -1j * np.array(X)

        # QM Part
        C = self.ctmqc_env['C'][irep, iatom]
        self.ctmqc_env['adPops'][irep] = calc_ad_pops(C, self.ctmqc_env)
        QM = qUt.calc_QM_FD(self.ctmqc_env, irep, iatom)
        adMom = qUt.calc_ad_mom(self.ctmqc_env, irep, iatom)

        tmp = 0.0
        for k in range(nstates):
            Ck = C[k]
            tmp += Ck * adMom[k]

        for l in range(nstates):
            X[l, l] += QM * (tmp - adMom[l])

        return np.array(X)

    def __RK4(self, coeff):
        """
        Will carry out the RK4 algorithm to propagate the coefficients
        """
        coeff = np.array(coeff)

        K1 = np.array(self.dTe * np.dot(np.array(self.X1), coeff))
        K2 = np.array(self.dTe * np.dot(np.array(self.X12), coeff + K1/2.))
        K3 = np.array(self.dTe * np.dot(np.array(self.X12), coeff + K2/2.))
        K4 = np.array(self.dTe * np.dot(np.array(self.X2), coeff + K3))
        Ktot = (1./6.) * (K1 + (2.*K2) + (2.*K3) + K4)
        coeff = coeff + Ktot

        return coeff

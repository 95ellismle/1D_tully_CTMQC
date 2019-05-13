#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:10 2019

@author: mellis
"""
import numpy as np

import hamiltonian as Ham


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


class elecProp(object):
    """
    Will handle the electronic propagation (make the X matrix and feed it into
    RK4)
    """
    def __init__(self, ctmqc_env):
        self.ctmqc_env = ctmqc_env
        self.dTe = self.ctmqc_env['dt'] / float(self.ctmqc_env['elec_steps'])

    def do_diab_prop(self, irep):
        """
        Will propagate the coefficients in the diabatic basis (without the
        diabatic NACE)
        """
        dx_E = (self.ctmqc_env['pos'] - self.ctmqc_env['pos_tm'])
        dx_E /= float(self.ctmqc_env['elec_steps'])

        self.X1 = self.makeX_diab(self.ctmqc_env['pos_tm'], irep)
        for Estep in range(self.ctmqc_env['elec_steps']):
            pos2 = self.ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
            pos3 = self.ctmqc_env['pos_tm'] + (Estep+1) * dx_E

            self.X12 = self.makeX_diab(pos2, irep)
            self.X2 = self.makeX_diab(pos3, irep)

            self.ctmqc_env['u'][irep] = self.__RK4(self.ctmqc_env['u'][irep])

            self.X1 = self.X2[:]

    def do_adiab_prop(self, irep):
        """
        Will actually carry out the propagation of the coefficients
        """
        dx_E = (self.ctmqc_env['pos'] - self.ctmqc_env['pos_tm'])
        dx_E /= float(self.ctmqc_env['elec_steps'])

        self.X1 = self.makeX_adiab(self.ctmqc_env['pos_tm'], irep)
        for Estep in range(self.ctmqc_env['elec_steps']):
            pos2 = self.ctmqc_env['pos_tm'] + (Estep + 0.5) * dx_E
            pos3 = self.ctmqc_env['pos_tm'] + (Estep+1) * dx_E

            self.X12 = self.makeX_adiab(pos2, irep)
            self.X2 = self.makeX_adiab(pos3, irep)

            self.ctmqc_env['C'][irep] = self.__RK4(self.ctmqc_env['C'][irep])

            self.X1 = self.X2[:]

    def makeX_diab(self, pos, irep):
        """
        Will make the diabatic X matrix
        """
        H = self.ctmqc_env['Hfunc'](pos[irep])

        return -1j * H

    def makeX_adiab(self, pos, irep):
        """
        Will make the adiabatic X matrix
        """
        nstates = self.ctmqc_env['nstate']
        X = np.zeros((nstates, nstates), dtype=complex)

        H = self.ctmqc_env['Hfunc'](pos)
        E, U = Ham.getEigProps(H, self.ctmqc_env)

        NACV = Ham.calcNACV(irep, self.ctmqc_env)
#        print(NACV, "\n")
        v = self.ctmqc_env['vel'][irep]

        # First part
        for l in range(nstates):
            X[l, l] = E[l]

        # Second part
        X += -1j * NACV * v[0]

        return -1j * X

    def __RK4(self, coeff):
        """
        Will carry out the RK4 algorithm to propagate the coefficients
        """
        K1 = np.array(self.dTe * np.dot(self.X1, coeff))[0]
        K2 = np.array(self.dTe * np.dot(self.X12, coeff + K1/2.))[0]
        K3 = np.array(self.dTe * np.dot(self.X12, coeff + K2/2.))[0]
        K4 = np.array(self.dTe * np.dot(self.X2, coeff + K3))[0]

        Ktot = (1./6.) * (K1 + (2.*K2) + (2.*K3) + K4)

        coeff = coeff + Ktot

        return coeff

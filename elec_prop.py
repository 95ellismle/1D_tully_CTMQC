#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:10 2019

@author: mellis
"""
import numpy as np

import hamiltonian as Ham


def calc_ad_pops(C, ctmqc_env):
    """
    Will calculate the adiabatic populations of all replicas
    """
    nstate = ctmqc_env['nstate']
    if len(C) != nstate or len(np.shape(C)) != 1:
        raise SystemExit("Incorrect Shape for adiab coeff in calc_ad_pops")
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

    def do_adiab_prop(self, irep):
        """
        Will actually carry out the propagation of the coefficients
        """
        pos_diff = (self.ctmqc_env['pos'] - self.ctmqc_env['pos_tm']) \
                 / float(self.ctmqc_env['elec_steps'])

        for Estep in range(self.ctmqc_env['elec_steps']):
            pos1 = self.ctmqc_env['pos_tm'] + Estep * pos_diff
            pos2 = self.ctmqc_env['pos_tm'] + (Estep + 0.5) * pos_diff
            pos3 = self.ctmqc_env['pos_tm'] + (Estep+1) * pos_diff
            
            X1 = self.makeX_adiab(pos1, irep)
            X12 = self.makeX_adiab(pos2, irep)
            X2 = self.makeX_adiab(pos3, irep)

    def makeX_adiab(self, pos, irep):
        """
        Will make the adiabatic X matrix
        """
        nstates = self.ctmqc_env['nstate']
        X = np.zeros((nstates, nstates))

        H = self.ctmqc_env['Hfunc'](pos)
        E, U = Ham.getEigProps(H, self.ctmqc_env)
        NACV = Ham.calcNACV(irep, self.ctmqc_env)
        v = self.ctmqc_env['vel'][irep]
        
        NACE

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:10 2019

@author: mellis
"""
import numpy as np


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
    return pops


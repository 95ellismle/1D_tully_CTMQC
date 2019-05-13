#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:25:43 2019

@author: oem
"""
import numpy as np

#import hamiltonian as Ham
import nucl_prop


def calc_ad_mom(ctmqc_env, irep):
    """
    Will calculate the adiabatic momenta (time-integrated adiab force)
    """
    pos = ctmqc_env['pos'][irep]
    ad_frc = nucl_prop.calc_ad_frc(pos, ctmqc_env)
    ad_mom = ctmqc_env['adMom'][irep]
    dt = ctmqc_env['dt']

    ad_mom += dt * ad_frc
    return ad_mom


def calc_Rvl(adPops, ctmqc_env, irep):
    """
    Will calculate the funny Rvl term in the Agostini, 17 paper
    """
    for I in range(ctmqc_env['nrep']):
        pos = ctmqc_env['pos'][I]


def calc_QM(ctmqc_env, adPops, irep):
    """
    Will calculate the quantum momentum.
    """
    calc_Rvl(adPops, ctmqc_env, irep)
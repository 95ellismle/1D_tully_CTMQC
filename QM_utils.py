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


def gaussian(RIv, RJv, sigma):
    """
    Will calculate the gaussian on the traj
    """
    pre_fact = (1. / (2 * np.pi * sigma**2)) ** (1/2)
    exponent = -((RIv - RJv)**2 / (2 * sigma**2))
    return pre_fact * np.exp(exponent)


def calc_WIJ(ctmqc_env, I):
    """
    Will calculate the WIJ term from Min, 17 SI
    """
    RIv = ctmqc_env['pos'][I]
    RJv = ctmqc_env['pos'][J]


def calc_QM(adPops, ctmqc_env, irep):
    """
    Will calculate the quantum momentum.
    """
    calc_WIJ()
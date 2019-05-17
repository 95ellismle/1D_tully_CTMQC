#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:25:43 2019

@author: oem
"""
import numpy as np
import scipy.integrate as integrate

import hamiltonian as Ham


def calc_ad_frc(pos, ctmqc_env):
    """
    Will calculate the forces from each adiabatic state (the grad E term)
    """
    dx = ctmqc_env['dx']
    H_xm = ctmqc_env['Hfunc'](pos - dx)
    H_x = ctmqc_env['Hfunc'](pos)
    H_xp = ctmqc_env['Hfunc'](pos + dx)
    allH = [H_xm, H_x, H_xp]
    allE = [Ham.getEigProps(H, ctmqc_env)[0] for H in allH]
    grad = np.array(np.gradient(allE, dx, axis=0))[2]
    return grad


def calc_ad_mom(ctmqc_env, irep):
    """
    Will calculate the adiabatic momenta (time-integrated adiab force)
    """
    pos = ctmqc_env['pos'][irep]
    ad_frc = calc_ad_frc(pos, ctmqc_env)
    ad_mom = ctmqc_env['adMom'][irep]
    dt = ctmqc_env['dt']

    ad_mom += dt * ad_frc
    return ad_mom


def gaussian(RIv, RJv, sigma):
    """
    Will calculate the gaussian on the traj

    This has been tested and is definitely normalised
    """
    pre_fact = (1. / (2 * np.pi * sigma**2)) ** (1/2)
    exponent = -((RIv - RJv)**2 / (2 * sigma**2))
    return pre_fact * np.exp(exponent)


def test_norm_gauss():
    """
    Will test whether the gaussian is normalised
    """
    x = np.arange(-10, 10, 0.01)
    y = gaussian(x, -2, 1)
    norm = integrate.simps(y, x)
    if abs(norm - 1) > 1e-9:
        raise SystemExit("Gaussian not normalised!")
    print("All is well with the gaussian")


def calc_nucl_dens_prime(ctmqc_env, I):
    """
    Will calculate the nuclear density summed over all replicas (not averaged).
    """
    nucl_dens = np.zeros(ctmqc_env['nrep'])

    RIv = ctmqc_env['pos'][I]
    for J in range(ctmqc_env['nrep']):
        RJv = ctmqc_env['pos'][J]
        nucl_dens[J] = gaussian(RIv, RJv, ctmqc_env['sigma'])

    return np.sum(nucl_dens)


def calc_QM(adPops, ctmqc_env, I):
    """
    Will calculate the quantum momentum.
    """
    WIJ = np.zeros((ctmqc_env['nrep'], ctmqc_env['nrep']))
    nucl_dens = calc_nucl_dens_prime(ctmqc_env, I)
    RIv = ctmqc_env['pos'][I]
    QM = 0.0

    # Calc WIJ
    for J in range(ctmqc_env['nrep']):
        RJv = ctmqc_env['pos'][J]

        WIJ[I, J] = gaussian(RIv, RJv, ctmqc_env['sigma'])
        WIJ[I, J] /= (2. * (ctmqc_env['sigma']**2) * nucl_dens)

    # Calc QM
    for J in range(ctmqc_env['nrep']):
        RJv = ctmqc_env['pos'][J]

        QM += WIJ[I, J] * (RIv - RJv)
        QM /= ctmqc_env['mass'][0]

    return QM

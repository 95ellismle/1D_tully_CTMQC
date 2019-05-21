#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:25:43 2019

@author: oem
"""
import numpy as np
import scipy.integrate as integrate
import random as rd

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


def calc_ad_mom(ctmqc_env, irep, v):
    """
    Will calculate the adiabatic momenta (time-integrated adiab force)
    """
    pos = ctmqc_env['pos'][irep, v]
    ad_frc = calc_ad_frc(pos, ctmqc_env)
    ad_mom = ctmqc_env['adMom'][irep, v]
    dt = ctmqc_env['dt']

    ad_mom += dt * ad_frc
    return ad_mom


def gaussian(RIv, RJv, sigma):
    """
    Will calculate the gaussian on the traj

    This has been tested and is definitely normalised
    """
    sig2 = sigma**2
    pre_fact = (sig2 * 2 * np.pi)**(-0.5)
    exponent = -((RIv - RJv)**2 / (2 * sig2))
    return pre_fact * np.exp(exponent)


def test_norm_gauss():
    """
    Will test whether the gaussian is normalised
    """
    x = np.arange(-10, 10, 0.01)
    y = gaussian(x, -1, abs(rd.gauss(0.5, 0.1)))
    norm = integrate.simps(y, x)
    if abs(norm - 1) > 1e-9:
        raise SystemExit("Gaussian not normalised!")


def calc_nucl_dens_PP(R, sigma):
    """
    Will calculate the nuclear density post-production

    Inputs:
        * R => the positions for 1 step <np.array (nrep, natom)>
        * sigma => the nuclear widths for 1 step <np.array (nrep, natom)>
    """
    nRep, nAtom = np.shape(R)
    nuclDens = np.zeros(nRep)

    for I in range(nRep):
        RIv = R[I, 0]
        allGauss = [gaussian(RIv, RJv, sigma[I, 0])
                    for RJv in R[:, 0]]
        nuclDens[I] = np.mean(allGauss)

    return nuclDens, R[:, 0]


def calc_nucl_dens(RIv, RJv, sigma):
    """
    Will calculate the nuclear density on replica I (for 1 atom)

    Inputs:
        * RIv => <float> The pos of replica I
        * RJv => <array> The pos of replicas J (all rep)
        * sigma => the width of the gaussian on replica J
    """
    allGauss = [gaussian(RIv, RJ, sig) for sig, RJ in zip(sigma, RJv)]
    return np.mean(allGauss)


def calc_QM(adPops, ctmqc_env, I, v):
    """
    Will calculate the quantum momentum (only for 1 atom currently)
    """
    RIv = ctmqc_env['pos'][I, v]
    nrep = len(ctmqc_env['pos'][:, v])

    nuclDens_xm = calc_nucl_dens(RIv - ctmqc_env['dx'],
                                 ctmqc_env['pos'][:, v],
                                 np.ones(nrep)*ctmqc_env['sigma'])

    nuclDens = calc_nucl_dens(RIv,
                              ctmqc_env['pos'][:, v],
                              np.ones(nrep)*ctmqc_env['sigma'])

    nuclDens_xp = calc_nucl_dens(RIv + ctmqc_env['dx'],
                                 ctmqc_env['pos'][:, v],
                                 np.ones(nrep)*ctmqc_env['sigma'])

    gradNuclDens = np.gradient([nuclDens_xm, nuclDens, nuclDens_xp],
                               ctmqc_env['dx'])[1]
    return gradNuclDens/(2*nuclDens)
#
#    raise SystemExit("BREAK")
#    WIJ = np.zeros(ctmqc_env['nrep'])  # Only calc WIJ for rep I
#    allGauss = [gaussian(RIv, RJv, ctmqc_env['sigma'])
#                for RJv in ctmqc_env['pos'][:, v]]
#
#    QM = 0.0
#
#    # Calc WIJ
#    sigma2 = ctmqc_env['sigma']**2
#    WIJ = allGauss / (2. * sigma2 * np.sum(allGauss))
#
#    # Calc QM
#    for J in range(ctmqc_env['nrep']):
#        RJv = ctmqc_env['pos'][J]
#
#        QM += WIJ[J] * (RIv - RJv)
#        QM /= ctmqc_env['mass'][0]
#
#    return QM 


test_norm_gauss()


'''
def calc_prod_gauss(RIv)


def calc_Qlk(adPops, adMom, ctmqc_env, I, v):
    """
    Will calculate the Qlk quantum momentum.
    """
    pass
'''

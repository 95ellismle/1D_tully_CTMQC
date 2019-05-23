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
    gradE = np.array(np.gradient(allE, dx, axis=0))[2]
    return -gradE


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
    Will test whether the gaussian is normalised by checking 50 different
    random gaussians.
    """
    x = np.arange(-10, 10, 0.001)
    y = [gaussian(x, -1, abs(rd.gauss(0.5, 0.3))+0.05) for i in range(50)]
    norms = [integrate.simps(i, x) for i in y]
    if any(abs(norm - 1) > 1e-9 for norm in norms):
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


def calc_nucl_dens(RIv, v, ctmqc_env):
    """
    Will calculate the nuclear density on replica I (for 1 atom)

    Inputs:
        * I => <int> Replica Index I
        * v => <int> Atom index v
        * sigma => the width of the gaussian on replica J atom v
        * ctmqc_env => the ctmqc environment
    """
    RJv = ctmqc_env['pos'][:, v]
    sigma = ctmqc_env['sigma'][:, v]
    allGauss = [gaussian(RIv, RJ, sig) for sig, RJ in zip(sigma, RJv)]
    return np.mean(allGauss)


def calc_sigma(ctmqc_env, I, v):
    """
    Will calculate the value of sigma used in constructing the nuclear density
    """
    cnst = ctmqc_env['const']
    sigma_tm = ctmqc_env['sigma_tm'][I, v]
    cutoff_rad = cnst * sigma_tm
    sig_thresh = cnst/ctmqc_env['nrep'] * np.min(ctmqc_env['sigma_tm'])

    distances = ctmqc_env['pos'] - ctmqc_env['pos'][I, v]
    new_var = np.std(distances[distances < cutoff_rad])

    if new_var < sig_thresh:
        new_var = sig_thresh
    ctmqc_env['sigma'][I, v] = new_var


def calc_QM_FD(ctmqc_env, I, v):
    """
    Will calculate the quantum momentum (only for 1 atom currently)
    """
#    calc_sigma(ctmqc_env, I, v)
    RIv = ctmqc_env['pos'][I, v]
    dx = ctmqc_env['dx']

    nuclDens_xm = calc_nucl_dens(RIv - dx, v, ctmqc_env)
    nuclDens = calc_nucl_dens(RIv, v, ctmqc_env)
    nuclDens_xp = calc_nucl_dens(RIv, v, ctmqc_env)

    gradNuclDens = np.gradient([nuclDens_xm, nuclDens, nuclDens_xp],
                               ctmqc_env['dx'])[1]
    if nuclDens < 1e-12:
        return 0
    return -gradNuclDens/(2*nuclDens)


def calc_QM_analytic(ctmqc_env, I, v):
    """
    Will use the analytic formula provided in SI to calculate the QM.
    """
#    calc_sigma(ctmqc_env, I, v)
    RIv = ctmqc_env['pos'][I, v]
    WIJ = np.zeros(ctmqc_env['nrep'])  # Only calc WIJ for rep I
    allGauss = [gaussian(RIv, RJv, sig)
                for (RJv, sig) in zip(ctmqc_env['pos'][:, v],
                                      ctmqc_env['sigma'][:, v])]
    QM = 0.0

    # Calc WIJ
    sigma2 = ctmqc_env['sigma'][:, v]**2
    WIJ = allGauss / (2. * sigma2 * np.sum(allGauss))

    # Calc QM
    for J in range(ctmqc_env['nrep']):
        RJv = ctmqc_env['pos'][J, v]

        QM += WIJ[J] * (RIv - RJv)
        QM /= ctmqc_env['mass'][0]

    return QM


test_norm_gauss()


'''
def calc_prod_gauss(RIv)


def calc_Qlk(adPops, adMom, ctmqc_env, I, v):
    """
    Will calculate the Qlk quantum momentum.
    """
    pass
'''

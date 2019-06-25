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


def calc_gradE(pos, ctmqc_env):
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
    return gradE


def calc_ad_mom(ctmqc_env, irep, v, ad_frc=False):
    """
    Will calculate the adiabatic momenta (time-integrated adiab force)
    """
    if ad_frc is False:
        pos = ctmqc_env['pos'][irep, v]
        ad_frc = -calc_gradE(pos, ctmqc_env)

    ad_mom = ctmqc_env['adMom'][irep, v]
    dt = ctmqc_env['dt']

    ad_mom += dt * ad_frc
#    print(ad_mom)
    return ad_mom


def gaussian(RIv, RJv, sigma):
    """
    Will calculate the gaussian on the traj

    This has been tested and is definitely normalised
    """
    sig2 = sigma**2

    prefact = (sig2 * 2 * np.pi)**(-0.5)
    exponent = -((RIv - RJv)**2 / (2 * sig2))

    return prefact * np.exp(exponent)


def test_norm_gauss():
    """
    Will test whether the gaussian is normalised by checking 50 different
    random gaussians.
    """
    x = np.arange(-13, 13, 0.001)
    y = [gaussian(x, -1, abs(rd.gauss(0.5, 0.3))+0.05) for i in range(50)]
    norms = [integrate.simps(i, x) for i in y]
    if any(abs(norm - 1) > 1e-9 for norm in norms):
        print(norms)
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


def calc_sigma(ctmqc_env):
    """
    Will calculate the value of sigma used in constructing the nuclear density.

    This algorithm doesn't seem to work -it gives discontinuous sigma and sigma
    seems to blow up. To fix discontinuities a weighted stddev might work.
    """
    for I in range(ctmqc_env['nrep']):
        for v in range(ctmqc_env['natom']):
            cnst = ctmqc_env['const']
            sigma_tm = ctmqc_env['sigma_tm'][I, v]
            cutoff_rad = cnst * sigma_tm
            sig_thresh = cnst/ctmqc_env['nrep'] * np.min(ctmqc_env['sigma_tm'])

            distances = ctmqc_env['pos'] - ctmqc_env['pos'][I, v]
            distMask = distances < cutoff_rad
            if any(distMask):
                new_var = np.std(distances[distances < cutoff_rad])
            else:
                print(cutoff_rad)
                new_var = sig_thresh
            ctmqc_env['sigma'][I, v] = new_var


def calc_QM_FD(ctmqc_env):
    """
    Will calculate the quantum momentum (only for 1 atom currently)
    """
    nRep, nAtom = ctmqc_env['nrep'], ctmqc_env['natom']
    QM = np.zeros((nRep, nAtom))
    for I in range(nRep):
        for v in range(nAtom):
            RIv = ctmqc_env['pos'][I, v]
            dx = ctmqc_env['dx']
        
            nuclDens_xm = calc_nucl_dens(RIv - dx, v, ctmqc_env)
            nuclDens = calc_nucl_dens(RIv, v, ctmqc_env)
            nuclDens_xp = calc_nucl_dens(RIv + dx, v, ctmqc_env)
        
            gradNuclDens = np.gradient([nuclDens_xm, nuclDens, nuclDens_xp],
                                       dx)[2]
            if nuclDens < 1e-12:
                return 0
        
            QM[I, v] = -gradNuclDens/(2*nuclDens)
        
    return QM / ctmqc_env['mass'][v]


def calc_QM_analytic(ctmqc_env):
    """
    Will use the analytic formula provided in SI to calculate the QM.
    """
    nRep, nAtom = ctmqc_env['nrep'], ctmqc_env['natom']
    QM = np.zeros((nRep, nAtom))
    for I in range(nRep):
        for v in range(nAtom):
            RIv = ctmqc_env['pos'][I, v]
            WIJ = np.zeros(ctmqc_env['nrep'])  # Only calc WIJ for rep I
            allGauss = [gaussian(RIv, RJv, sig)
                        for (RJv, sig) in zip(ctmqc_env['pos'][:, v],
                                              ctmqc_env['sigma'][:, v])]
            # Calc WIJ
            sigma2 = ctmqc_env['sigma'][:, v]**2
            WIJ = allGauss / (2. * sigma2 * np.sum(allGauss))
        
            # Calc QM
            QM[I, v] = np.sum(WIJ * (RIv - ctmqc_env['pos'][:, v]))
        
    return QM / ctmqc_env['mass'][v]


def calc_all_prod_gauss(ctmqc_env):
    """
    Will calculate the product of the gaussians in a more efficient way than
    simply brute forcing it.
    """
    nRep, nAtom = ctmqc_env['nrep'], ctmqc_env['natom']
    
    # We don't need the prefactor of (1/(2 pi))^{3/2} as it always cancels out
    prefact = np.prod(ctmqc_env['sigma']**(-1), axis=1)
    # Calculate the exponent
    exponent = np.zeros((nRep, nRep))
    for v in range(nAtom):
        for I in range(nRep):
            RIv = ctmqc_env['pos'][I, v]
            for J in range(nRep):
                RJv = ctmqc_env['pos'][J, v]
                sJv = ctmqc_env['sigma'][J, v]
                
                exponent[I, J] -= ( (RIv - RJv)**2 / (sJv**2))
    return np.exp(exponent * 0.5) * prefact


def calc_WIJ(ctmqc_env, reps_to_complete=False):
    """
    Will calculate alpha for all replicas and atoms
    """
    nRep, nAtom = ctmqc_env['nrep'], ctmqc_env['natom']
    WIJ = np.zeros((nRep, nRep, nAtom))
    allProdGauss_IJ = calc_all_prod_gauss(ctmqc_env)

    if reps_to_complete is not False:
        for I in reps_to_complete:
            for v in range(nAtom):
                # Calc WIJ and alpha
                sigma2 = ctmqc_env['sigma'][:, v]**2
                WIJ[I, :, v] = allProdGauss_IJ[I, :] \
                                / (sigma2 * np.sum(allProdGauss_IJ[I, :]))
        WIJ /= 2.
    else:
         for I in range(nRep):
            for v in range(nAtom):
                # Calc WIJ and alpha
                sigma2 = ctmqc_env['sigma'][:, v]**2
                WIJ[I, :, v] = allProdGauss_IJ[I, :] \
                                / (2. * sigma2 * np.sum(allProdGauss_IJ[I, :]))
    return WIJ


def calc_Qlk(ctmqc_env):
    """
    Will calculate the (effective) intercept for the Quantum Momentum based on
    the values in the ctmqc_env dict. If this spikes then the R0 will be used,
    if the Rlk isn't spiking then the Rlk will be used.
    """
    # Calculate Rlk -compare it to previous timestep Rlk
    nRep, nAtom, = ctmqc_env['nrep'], ctmqc_env['natom']
    nState = ctmqc_env['nstate']

    pops = ctmqc_env['adPops']
    reps_to_complete = np.arange(nRep)
    #reps_to_complete = [irep for irep, rep_pops in enumerate(pops[:, 0, :])
    #                    if all(state_pop < 0.995 for state_pop in rep_pops)]

    # Calculate WIJ and alpha
    WIJ = calc_WIJ(ctmqc_env, reps_to_complete)
    alpha = np.sum(WIJ, axis=1)
    ctmqc_env['alpha'] = alpha
    Ralpha = ctmqc_env['pos'] * alpha

    # Calculate all the Ylk
    f = ctmqc_env['adMom']
    Ylk = np.zeros((nRep, nAtom, nState, nState))
    for J in reps_to_complete:
        for v in range(nAtom):
            for l in range(nState):
                Cl = pops[J, v, l]
                fl = f[J, v, l]
                for k in range(l):
                    Ck = pops[J, v, k]
                    fk = f[J, v, k]
                    Ylk[J, v, l, k] = Ck * Cl * (fk - fl)
                    Ylk[J, v, k, l] = -Ylk[J, v, l, k]
    sum_Ylk = np.sum(Ylk, axis=0)  # sum over replicas
    # Calculate Qlk
    Qlk = np.zeros((nRep, nAtom, nState, nState))
    if abs(sum_Ylk[0, 0, 1]) > 1e-12:
        # Calculate the R0 (used if the Rlk spikes)
        RI0 = np.zeros((nRep, nAtom))
        for I in reps_to_complete:
            for v in range(nAtom):
                RI0[I, v] = np.dot(WIJ[I, :, v], ctmqc_env['pos'][:, v])

        # Calculate the Rlk
        Rlk = np.zeros((nAtom, nState, nState))
        for I in reps_to_complete:
            for v in range(nAtom):
                Rav = Ralpha[I, v]
                for l in range(nState):
                    for k in range(l):
                        Rlk[v, l, k] += Rav * (
                                     Ylk[I, v, l, k] / sum_Ylk[v, l, k])
                        Rlk[v, k, l] = Rlk[v, l, k]
            
        # Save the data
        ctmqc_env['Rlk'] = Rlk
        ctmqc_env['RI0'] = RI0

        # Calculate the Quantum Momentum
        maxRI0 = np.max(RI0[np.abs(RI0) > 0], axis=0)
        minRI0 = np.min(RI0[np.abs(RI0) > 0], axis=0)
        for I in reps_to_complete:
            for v in range(nAtom):
                for l in range(nState):
                    for k in range(nState):
                        if Rlk[v, l, k] > maxRI0 or Rlk[v, l, k] < minRI0:
                            R = RI0[I, v]
                        else:
                            R = Rlk[v, l, k]

                        Qlk[I, v, l, k] = Ralpha[I, v] - R

        # Divide by mass2
        for v in range(nAtom):
            Qlk[:, v, :, :] /= ctmqc_env['mass'][v]

    return Qlk



def test_QM_calc(ctmqc_env):
    """
    Will compare the output of the analytic QM calculation and the finite
    difference one.
    """
    allDiffs = calc_QM_analytic(ctmqc_env) - calc_QM_FD(ctmqc_env)
    allPos = ctmqc_env['pos']

    print("Avg Abs Diff = %.2g +/- %.2g" % (np.mean(allDiffs),
                                              np.std(allDiffs)))
    print("Max Abs Diff = %.2g" % np.max(np.abs(allDiffs)))
    print("Min Abs Diff = %.2g" % np.min(np.abs(allDiffs)))
    if np.max(np.abs(allDiffs)) > 1e-5:
        raise SystemExit("Analytic Quantum Momentum != Finite Difference")

    return allDiffs, allPos


#test_norm_gauss()

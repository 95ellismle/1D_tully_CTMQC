from __future__ import print_function
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:25:43 2019

@author: oem
"""
import numpy as np

import clustering as clust


def calc_ad_frc(pos, ctmqc_env):
    """
    Will calculate the forces from each adiabatic state (the grad E term)
    """
    dx = ctmqc_env['dx']
    H_x = ctmqc_env['Hfunc'](pos)
    H_xp = ctmqc_env['Hfunc'](pos + dx)
    E_xp = np.linalg.eigh(H_xp)[0]
    E_x = np.linalg.eigh(H_x)[0]
    gradE = -np.array(E_xp - E_x) / dx

    return gradE


def calc_ad_mom(ctmqc_env, irep, ad_frc=False):
    """
    Will calculate the adiabatic momenta (time-integrated adiab force)
    """
    if ad_frc is False:
        pos = ctmqc_env['pos'][irep]
        ad_frc = calc_ad_frc(pos, ctmqc_env)

    ad_mom = ctmqc_env['adMom'][irep]
    dt = ctmqc_env['dt']

    ad_mom += dt * ad_frc
#    print(ad_mom)
    return ad_mom


def smoothingFunc(x, ctmqc_env):
    """
    Will do the smoothing between points
    """
    dt = ctmqc_env['dt']
    pos = ctmqc_env['smoothInitT']
    finalPos = pos + (dt * ctmqc_env['nSmoothStep'])
    midPoint = (finalPos + pos) / 2.

    D = 0.5 * (ctmqc_env['currGoodPoint'] - ctmqc_env['lastGoodPoint'])
    W = ctmqc_env['nSmoothStep'] * dt * 0.3

    return ctmqc_env['lastGoodPoint'] + (np.tanh((x - midPoint) / W) + 1) * D


def do_Rlk_smoothing(effR, ctmqc_env):
    """
    Will carry out the smoothing between the Rlk and the R0 intercepts.
    """
    # Do the smoothing between Rl and Rlk
    ctmqc_env['currGoodPoint'] = effR
    # Boundary conditions
    #   Start spiking
    if ctmqc_env['isSpiking'] != ctmqc_env['prevSpike']:
        ctmqc_env['iSmoothStep']  = 1
        ctmqc_env['lastGoodPoint'] = ctmqc_env['effR']
        ctmqc_env['smoothInitT'] = ctmqc_env['dt'] * (ctmqc_env['iter'] - 1)
        effR = smoothingFunc(ctmqc_env['t'], ctmqc_env)
    #   End Spiking
    elif ctmqc_env['iSmoothStep'] == ctmqc_env['nSmoothStep']:
        ctmqc_env['iSmoothStep'] = -1

    # Smoothing between boundaries
    elif 0 < ctmqc_env['iSmoothStep'] and ctmqc_env['iSmoothStep'] < ctmqc_env['nSmoothStep']:
        effR = smoothingFunc(ctmqc_env['t'], ctmqc_env)
        ctmqc_env['iSmoothStep'] += 1

    return effR


def Lagrange_Extrapolation(arrayX, arrayY, x):
    """
    Will use Lagrange polynomials to extrapolate the data to estimate the Rlk
    after N steps (N depends on how big the Rlk array is).
    """
    result = 0.0
    k = len(arrayY) - 1
    for j, yj in enumerate(arrayY):
        lj = 1.0
        for m in range(k+1):
            if m == j: continue
            xm = arrayX[m]
            xj = arrayX[j]
            lj *= float(x - xm)/float(xj - xm)

        result += yj * lj

    return result


def Rlk_is_spiking(Rlk, runData):
    """
    Will determine whether the Rlk intercept term is spiking and if it is return True,
    else return False. Currently all this does is check that the Rlk isn't too far away
    from the Rl intercept.
    """
    ctmqc_env = runData.ctmqc_env
    if ctmqc_env['iter'] == 0 and not ctmqc_env['Rlk_smooth']: return False

    # Check whether the Rlk will be near 0 in the next N steps
    Nstep = 50 / ctmqc_env['dt']  # 50 atomic unit window
    if ctmqc_env['spike_region_count'] >= Nstep:
        ctmqc_env['poss_spike'] = False

    if not ctmqc_env['poss_spike']:
        Rlk_future = Nstep * (Rlk[0, 1] - ctmqc_env['Rlk_tm'][0, 1])
        if np.sign(Rlk_future) != np.sign(Rlk[0, 1]):
            ctmqc_env['spike_region_count'] = 0
            ctmqc_env['poss_spike'] = True
    else:
        ctmqc_env['spike_region_count'] += 1

    # Check whether the gradient of the Rlk is too high
    Rlk = Rlk[0, 1]
    gradRlk = (Rlk - ctmqc_env['Rlk_tm'][0, 1]) / ctmqc_env['dt']
    if abs(gradRlk) > ctmqc_env['gradTol']: # or Rlk > plus or Rlk < minus:
#       print(ctmqc_env['iter'] * ctmqc_env['dt'])
       return True
    return False


def get_goodR_RIO(ctmqc_env):
    """do_Rlk_smoothing
    Will return the 'goodR' term. That is the intercept that has the
    spikes removed. This will use the RI0 as a value to anchor the smoothing.
    """
    # Get alternative R
    ctmqc_env['altR'] = np.sum(ctmqc_env['WIJ'] * ctmqc_env['pos'], axis=1)

    # If it is spiking interpolate between the Rlk and RI0
    goodR = np.zeros((ctmqc_env['nstate'], ctmqc_env['nstate']))
    for l in range(ctmqc_env['nstate']):
        for k in range(l):
            goodR[l, k] = np.mean(ctmqc_env['altR'])
            goodR[k, l] = goodR[l, k]

    return goodR


def get_goodR_extrapolation(runData, Rlk):
    """
    Will return the 'goodR' term. That is the intercept that has the
    spikes removed. This will use the extrapolated Rlk to anchor the smoothing.
    """
    ctmqc_env = runData.ctmqc_env
    if ctmqc_env['isSpiking']:
        ctmqc_env['extrapCount'] += 1
    else:
        ctmqc_env['extrapCount'] = 0

    order = ctmqc_env['polynomial_order']
    backStep = int(10 // ctmqc_env['dt'])
    indS = ctmqc_env['iter'] - order + 1 - ctmqc_env['extrapCount'] - backStep
    indE = ctmqc_env['iter']  + 1 - ctmqc_env['extrapCount'] - backStep

    # Create the alternative R
    if ctmqc_env['iter'] < order: return False
    effR = Lagrange_Extrapolation(runData.allt[indS:indE],
                                  runData.allRlk[indS:indE],
                                  ctmqc_env['t'] + ctmqc_env['dt'])

    return effR


def get_goodR_LGP(runData):
    """
    Will return the 'goodR' term. That is the intercept that has the
    spikes removed. This will use the last good Rlk as a value to anchor the
    smoothing.
    """
    # Get alternative R
    ctmqc_env = runData.ctmqc_env
    if ctmqc_env['prevSpike'] != ctmqc_env['isSpiking']:
        ctmqc_env['bob'] = runData.allRlk[ctmqc_env['iter'] - 3]

    # If it is spiking interpolate between the Rlk and RI0
    goodR = ctmqc_env['bob']

    return goodR


def get_effective_R(runData, Rlk):
    """
    Will get the effective R using one of the alternative R
    """
    ctmqc_env = runData.ctmqc_env
    ctmqc_env['isSpiking'] = Rlk_is_spiking(Rlk, runData)
#    ctmqc_env['isSpiking'] = True

    effR = Rlk
    if ctmqc_env['isSpiking']:
        if ctmqc_env['Rlk_smooth'] == 'RI0':
            effR = get_goodR_RIO(ctmqc_env)

        elif 'extrapolation' in ctmqc_env['Rlk_smooth']:
            effR = get_goodR_extrapolation(runData, Rlk)

        elif ctmqc_env['Rlk_smooth'] == 'ehrenfest':
            effR = False

        elif ctmqc_env['Rlk_smooth'] == 'LGP':
            effR = get_goodR_LGP(runData)

        elif ctmqc_env['Rlk_smooth'].strip() == '':
            effR = Rlk

        else:
            print("I don't recognise the Rlk smoothing method chosen.")
            print("If no smoothing is required set the 'nSmoothStep' = 0.")
            print("and the 'Rlk_smooth' = ''.")
            raise SystemExit("No Rlk Smooth Method Inputted")

    if ctmqc_env['nSmoothStep'] > 0:  effR = do_Rlk_smoothing(effR, ctmqc_env)

    ctmqc_env['prevSpike'] = ctmqc_env['isSpiking']
    return effR


def calc_Qlk_2state(ctmqc_env):
    """
    Will calculate the quantum momentum for 2 states exactly as given in
    Frederica's paper.
    """
    nState, nRep = ctmqc_env['nstate'], ctmqc_env['nrep']
    pops = ctmqc_env['adPops']
    ctmqc_env['QM_reps'] = np.arange(nRep)# [irep for irep, rep_pops in enumerate(pops)
                           #if all(state_pop < 0.995 for state_pop in rep_pops)]
    calc_sigmal(ctmqc_env)
    #ctmqc_env['sigmal'][:] = 0.25

    alpha = np.nan_to_num(np.sum(ctmqc_env['adPops'][0] / ctmqc_env['sigmal']))
    ctmqc_env['alphal'] = alpha

    rho11 = pops[:, 0]
    rho22 = pops[:, 1]
    f1 = ctmqc_env['adMom'][:, 0]
    f2 = ctmqc_env['adMom'][:, 1]
    allBottomRlk = rho11 * rho22 * (f1 - f2)
    Rlk = np.sum(ctmqc_env['pos'] * (allBottomRlk / np.sum(allBottomRlk)))
    ctmqc_env['Rlk'][0, 1] = Rlk
    ctmqc_env['Rlk'][1, 0] = Rlk

    effR = get_goodR_RIO(ctmqc_env)

    Qlk = np.zeros((nRep, nState, nState))
    for I in range(nRep):
        Qlk[I, 0, 1] = ctmqc_env['pos'][I] - effR[0, 1]
        Qlk[I, 1, 0] = ctmqc_env['pos'][I] - effR[1, 0]
    Qlk = np.nan_to_num(Qlk)
    Qlk = alpha * Qlk / ctmqc_env['mass']

    return Qlk


def calc_sigmal(ctmqc_env):
    """
    Will calculate the state dependant sigma l term.
    """
    nState = ctmqc_env['nstate']
    nRep = ctmqc_env['nrep']
    pops = ctmqc_env['adPops']
    sumPops = np.sum(ctmqc_env['adPops'], axis=0)

    # First  get R_l
    sumInvPops = np.nan_to_num(1. / sumPops)
    Rl = np.zeros(nState)
    for irep in range(nRep):
        # implicit iteration over l
        Rl += ctmqc_env['pos'][irep] * pops[irep] * sumInvPops

    # Now get sigma_l
    sigmal = np.zeros(nState)
    for irep in range(nRep):
        # implicit iteration over l
        squaredDist = (ctmqc_env['pos'][irep] - Rl)**2
        sigmal += squaredDist * pops[irep] * sumInvPops

    ctmqc_env['sigmal'] = sigmal
    ctmqc_env['Rl'] = Rl

    return sigmal


def calc_all_prod_gauss(ctmqc_env, reps_to_do=False):
    """
    Will calculate the product of the gaussians in a more efficient way than
    simply brute forcing it.
    """
    nRep = ctmqc_env['nrep']

    # We don't need the prefactor of (1/(2 pi))^{3/2} as it always cancels out
    prefact = ctmqc_env['sigma']**(-1)
    # Calculate the exponent
    exponent = np.zeros((nRep, nRep))
    if reps_to_do is not False:
        for I in reps_to_do:
            RIv = ctmqc_env['pos'][I]
            for J in range(nRep):
                RJv = ctmqc_env['pos'][J]
                sJv = ctmqc_env['sigma'][J]

                exponent[I, J] -= ( (RIv - RJv)**2 / (sJv**2))
    else:
        for I in range(ctmqc_env['nrep']):
            RIv = ctmqc_env['pos'][I]
            for J in range(nRep):
                RJv = ctmqc_env['pos'][J]
                sJv = ctmqc_env['sigma'][J]

                exponent[I, J] -= ( (RIv - RJv)**2 / (sJv**2))
    return np.exp(exponent * 0.5) * prefact


def calc_WIJ(ctmqc_env, reps_to_complete=False):
    """
    Will calculate alpha for all replicas and atoms
    """
    nRep = ctmqc_env['nrep']
    WIJ = np.zeros((nRep, nRep))
    allProdGauss_IJ = calc_all_prod_gauss(ctmqc_env, reps_to_complete)

    if reps_to_complete is not False:
        for I in reps_to_complete:
            # Calc WIJ and alpha
            sigma2 = ctmqc_env['sigma']**2
            WIJ[I, :] = allProdGauss_IJ[I, :] \
                            / (sigma2 * np.sum(allProdGauss_IJ[I, :]))
            WIJ[I, :] /= 2
    else:
        for I in range(ctmqc_env['nrep']):
            # Calc WIJ and alpha
            sigma2 = ctmqc_env['sigma']**2
            WIJ[I, :] = allProdGauss_IJ[I, :] \
                            / (sigma2 * np.sum(allProdGauss_IJ[I, :]))
            WIJ[I, :] /= 2

    return WIJ


def calc_Ylk(ctmqc_env):
    """
    Will calculate the Ylk value that appears in the Rlk quantity
    """
    Ylk = np.zeros((ctmqc_env['nrep'],
                    ctmqc_env['nstate'],
                    ctmqc_env['nstate']))

    # Can eventually use antisymmtery for this.
    for I in range(ctmqc_env['nrep']):
        for l in range(ctmqc_env['nstate']):
            for k in range(l):
                Clk = ctmqc_env['adPops'][I, l] * ctmqc_env['adPops'][I, k]
                fl = ctmqc_env['adMom'][I, l]
                fk = ctmqc_env['adMom'][I, k]

                Ylk[I, l, k] = Clk  * (fk - fl)
                Ylk[I, k, l] = -Ylk[I, l, k]
    return Ylk


def calc_Rlk(ctmqc_env, reps_to_do=False):
    """
    Will calculate the pair-wise state dependence intercept used in the
    calculation of Qlk
    """
    Ylk = calc_Ylk(ctmqc_env)
    summed_Ylk = np.sum(Ylk, axis=0)

    Rlk = np.zeros((ctmqc_env['nstate'], ctmqc_env['nstate']))
#    new_reps_to_do = reps_to_do[:]
    if reps_to_do is not False:
        for l in range(ctmqc_env['nstate']):
            for k in range(ctmqc_env['nstate']):
                for I in reps_to_do:
                    RI = ctmqc_env['pos'][I]
#                    if Ylk[I, l, k] <= 1e-8:
#                        new_reps_to_do = new_reps_to_do[new_reps_to_do != I]
#                        print(new_reps_to_do)
#                        continue
                    Ylk_sum = summed_Ylk[l, k]
                    if summed_Ylk[l, k] != 0:
                        Rlk[l, k] += (RI * ctmqc_env['alpha'][I]
                                         * Ylk[I, l, k]) / Ylk_sum
    else:
        for l in range(ctmqc_env['nstate']):
            for k in range(ctmqc_env['nstate']):
                for I in range(ctmqc_env['nrep']):
                    RI = ctmqc_env['pos'][I]
                    Ylk_sum = summed_Ylk[l, k]
                    if summed_Ylk[l, k] != 0:
                        Rlk[l, k] += (RI * ctmqc_env['alpha'][I]
                                         * Ylk[I, l, k]) / Ylk_sum
#    print(new_reps_to_do)
#    raise SystemExit("BREAK")
    return Rlk


def calc_Gossel_sigma(ctmqc_env):
    """
    Will calculate the sigma parameter as laid out in Gossell, 18.
    """
    minSig = np.min(ctmqc_env['sigma'])

    clusterPos, clusterInds = clust.getClusters(ctmqc_env['pos'], 0.7, True)

    for I in range(ctmqc_env['nrep']):
        clustID = clust.getClustID(clusterInds, I)
        distances = np.abs(ctmqc_env['pos'][I] - clusterPos[clustID])
        multiplier = float(ctmqc_env['const']) / float(len(distances))

#        avgD = np.mean(distances)
#        Dsquared = np.mean(distances**2)
#        ctmqc_env['sigma'][I] = np.sqrt(Dsquared - (avgD**2)) * multiplier

        ctmqc_env['sigma'][I] = np.std(distances) * multiplier

    mask = ctmqc_env['sigma'] < minSig
    if sum(mask):
        ctmqc_env['sigma'][mask] = minSig


def calc_Gossel_sigma_with_clusters(ctmqc_env):
    """
    Will calculate the sigma parameter as laid out in Gossell, 18.
    """
    minSig = np.min(ctmqc_env['sigma'])

    clusterPos, clusterInds = clust.getClusters(ctmqc_env['pos'], 0.7, True)

    for I in range(ctmqc_env['nrep']):
        clustID = clust.getClustID(clusterInds, I)
        distances = np.abs(ctmqc_env['pos'][I] - clusterPos[clustID])
        multiplier = float(ctmqc_env['const']) / float(len(distances))

#        avgD = np.mean(distances)
#        Dsquared = np.mean(distances**2)
#        ctmqc_env['sigma'][I] = np.sqrt(Dsquared - (avgD**2)) * multiplier

        ctmqc_env['sigma'][I] = np.std(distances) * multiplier

    mask = ctmqc_env['sigma'] < minSig
    if sum(mask):
        ctmqc_env['sigma'][mask] = minSig


#    print(ctmqc_env['sigma'])
#    raise SystemExit("BREAK")


def calc_deBroglie_sigma(ctmqc_env):
    """
    Will calculate sigma from 50 * the De-Broglie wavelength.
    """
    ctmqc_env['sigma'] = 60 / (2 * np.pi * ctmqc_env['mass'] * abs(ctmqc_env['vel']))


def calc_Qlk_Min17_opt(runData):
    """
    Will calculate the quantum momentum as written in Min, 17.
    """
    ctmqc_env = runData.ctmqc_env
#
#    for I in range(ctmqc_env['nrep']):
#        pops = ctmqc_env['adPops'][I, 0]
#        ctmqc_env['sigma'][I] = 2 - (np.exp(-(pops-0.5)**2/0.08) * 1.9)
#    ctmqc_env['sigma'] = 0.2
    if ctmqc_env['do_sigma_calc'].lower() == 'gossel':
        calc_Gossel_sigma(ctmqc_env)
    elif ctmqc_env['do_sigma_calc'].lower() == 'de-broglie':
        calc_deBroglie_sigma(ctmqc_env)
    elif ctmqc_env['do_sigma_calc'].lower() == 'gossel_clusters':
        calc_Gossel_sigma_with_clusters(ctmqc_env)
    elif ctmqc_env['do_sigma_calc'].lower() == 'no':
        pass
    else:
        print("I don't know how to treat the sigma parameter")
        print("Options are:\n\t* 'Gossel'\n\t* 'De-Broglie'\n\t* 'No'")
        raise SystemExit("Unkown Input")

    threshold = 0.995
    mask = [not any(i) for i in runData.ctmqc_env['adPops'] > threshold]
    reps_to_do = np.arange(ctmqc_env['nrep'])[mask]

    # Now calculate intercept
    Rlk = calc_Rlk(ctmqc_env, reps_to_do)

    # Calculate slope
    ctmqc_env['WIJ'] = calc_WIJ(ctmqc_env, reps_to_do)
    ctmqc_env['alpha'] = np.sum(ctmqc_env['WIJ'], axis=1)



    # Smooth out the intercept
    effR = get_effective_R(runData, Rlk)

    ctmqc_env['Rlk'] = Rlk
    ctmqc_env['effR'] = effR

    # Finally calculate Qlk
    Qlk = np.zeros((ctmqc_env['nrep'],
                    ctmqc_env['nstate'],
                    ctmqc_env['nstate']))
    if effR is False: return Qlk

    for I in reps_to_do:
        Qlk[I, :, :] = (ctmqc_env['alpha'][I] * ctmqc_env['pos'][I]) - effR
        for l in range(ctmqc_env['nstate']):
            Qlk[I, l, l] = 0.0

    # Only for 2 states
    if np.any(Qlk[:, 0, 1] != Qlk[:, 1, 0]):
        print(Qlk)
        raise SystemExit("Qlk not symmetric!")

    Qlk /= ctmqc_env['mass']
#    print(Qlk)
#    print("Qlk = ", Qlk)
    return Qlk
#test_norm_gauss()
















'''
            ########################
            # Old Unused Code Repo #
            ########################









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
        * R => the positions for 1 step <np.array (nrep)>
        * sigma => the nuclear widths for 1 step <np.array (nrep)>
    """
    nRep = np.shape(R)
    nuclDens = np.zeros(nRep)

    for I in range(nRep):
        RIv = R[I]
        allGauss = [gaussian(RIv, RJv, sigma[I])
                    for RJv in R]
        nuclDens[I] = np.mean(allGauss)

    return nuclDens, R


def calc_nucl_dens(RIv, ctmqc_env):
    """
    Will calculate the nuclear density on replica I (for 1 atom)

    Inputs:
        * I => <int> Replica Index I
        * v => <int> Atom index v
        * sigma => the width of the gaussian on replica J atom v
        * ctmqc_env => the ctmqc environment
    """
    RJv = ctmqc_env['pos']
    sigma = ctmqc_env['sigma']
    allGauss = [gaussian(RIv, RJ, sig) for sig, RJ in zip(sigma, RJv)]
    return np.mean(allGauss)


def calc_sigma(ctmqc_env):
    """
    Will calculate the value of sigma used in constructing the nuclear density.

    This algorithm doesn't seem to work -it gives discontinuous sigma and sigma
    seems to blow up. To fix discontinuities a weighted stddev might work.
    """
    for I in range(ctmqc_env['nrep']):
        cnst = ctmqc_env['const']
        sigma_tm = ctmqc_env['sigma_tm'][I]
        cutoff_rad = cnst * sigma_tm
        sig_thresh = cnst/ctmqc_env['nrep'] * np.sqrt(2)

        distances = ctmqc_env['pos'] - ctmqc_env['pos'][I]
        distMask = distances < cutoff_rad
        if any(distMask):
            new_var = np.std(distances[distances < cutoff_rad])
        else:
            print(cutoff_rad)
            new_var = sig_thresh
        ctmqc_env['sigma'][I] = new_var


def calc_QM_FD(ctmqc_env):
    """
    Will calculate the quantum momentum (only for 1 atom currently)
    """
    nRep = ctmqc_env['nrep']
    QM = np.zeros(nRep)
    for I in range(nRep):
        RIv = ctmqc_env['pos'][I]
        dx = ctmqc_env['dx']

        nuclDens_xm = calc_nucl_dens(RIv - dx, ctmqc_env)
        nuclDens = calc_nucl_dens(RIv, ctmqc_env)
        nuclDens_xp = calc_nucl_dens(RIv + dx, ctmqc_env)

        gradNuclDens = np.gradient([nuclDens_xm, nuclDens, nuclDens_xp],
                                   dx)[2]
        if nuclDens < 1e-12:
            return 0

        QM[I] = -gradNuclDens/(2*nuclDens)

    return QM / ctmqc_env['mass']


def calc_QM_analytic(ctmqc_env):
    """
    Will use the analytic formula provided in SI to calculate the QM.
    """
    nRep = ctmqc_env['nrep']
    QM = np.zeros(nRep)
    for I in range(nRep):
        RIv = ctmqc_env['pos'][I]
        WIJ = np.zeros(ctmqc_env['nrep'])  # Only calc WIJ for rep I
        allGauss = [gaussian(RIv, RJv, sig)
                    for (RJv, sig) in zip(ctmqc_env['pos'],
                                          ctmqc_env['sigma'])]
        # Calc WIJ
        sigma2 = ctmqc_env['sigma']**2
        WIJ = allGauss / (2. * sigma2 * np.sum(allGauss))

        # Calc QM
        QM[I] = np.sum(WIJ * (RIv - ctmqc_env['pos']))

    return QM / ctmqc_env['mass']






def calc_Qlk(ctmqc_env):
    """
    Will calculate the (effective) intercept for the Quantum Momentum based on
    the values in the ctmqc_env dict. If this spikes then the R0 will be used,
    if the Rlk isn't spiking then the Rlk will be used.
    """
    # Calculate Rlk -compare it to previous timestep Rlk
    nRep = ctmqc_env['nrep']
    nState = ctmqc_env['nstate']

    pops = ctmqc_env['adPops']
#    reps_to_complete = np.arange(nRep)
    ctmqc_env['QM_reps'] = [irep for irep, rep_pops in enumerate(pops[:, :])
                            if all(state_pop < 0.995 for state_pop in rep_pops)]
#    calc_sigmal(ctmqc_env)
#    sig = np.mean(ctmqc_env['sigmal'])
#    if sig < 0.1:
#        sig = 0.1
#    ctmqc_env['sigma'][:] = sig

#    print(ctmqc_env['sigma'])

    # Calculate WIJ and alpha
    WIJ = calc_WIJ(ctmqc_env, ctmqc_env['QM_reps'])
    alpha = np.sum(WIJ, axis=1)
    ctmqc_env['alpha'] = alpha
    Ralpha = ctmqc_env['pos'] * alpha

    # Calculate all the Ylk
    f = ctmqc_env['adMom']
    Ylk = np.zeros((nRep, nState, nState))
    for J in ctmqc_env['QM_reps']:
        for l in range(nState):
            Cl = pops[J, l]
            fl = f[J, l]
            for k in range(l):
                Ck = pops[J, k]
                fk = f[J, k]
                Ylk[J, l, k] = Ck * Cl * (fk - fl)
                Ylk[J, k, l] = -Ylk[J, l, k]
    sum_Ylk = np.sum(Ylk, axis=0)  # sum over replicas
    # Calculate Qlk
    Qlk = np.zeros((nRep, nState, nState))
    if abs(sum_Ylk[0, 1]) > 1e-12:
        # Calculate the R0 (used if the Rlk spikes)
        RI0 = np.zeros((nRep))
        for I in ctmqc_env['QM_reps']:
            RI0[I] = np.dot(WIJ[I, :], ctmqc_env['pos'][:])

        # Calculate the Rlk
        Rlk = np.zeros((nState, nState))
        for I in ctmqc_env['QM_reps']:
            Rav = Ralpha[I]
            for l in range(nState):
                for k in range(l):
                    Rlk[l, k] += Rav * (
                                 Ylk[I, l, k] / sum_Ylk[l, k])
                    Rlk[k, l] = Rlk[l, k]

        # Save the data
        ctmqc_env['Rlk'] = Rlk
        ctmqc_env['RI0'] = RI0

        # Calculate the Quantum Momentum
        maxRI0 = np.max(RI0[np.abs(RI0) > 0], axis=0)
        minRI0 = np.min(RI0[np.abs(RI0) > 0], axis=0)
        for I in ctmqc_env['QM_reps']:
            R = np.zeros((nState, nState))
            for l in range(nState):
                for k in range(l):
                    if Rlk[l, k] > maxRI0 or Rlk[l, k] < minRI0:
#                        Qlk[I] = ctmqc_env['Qlk_tm'][I] * ctmqc_env['mass']
                        R[l, k] = RI0[I]
                        R[k, l] = RI0[I]
                    else:
                        R[l, k] = Rlk[l, k]
                        R[l, k] = Rlk[k, l]

            Qlk[I, :, :] = Ralpha[I] - R

            ctmqc_env['EffR'][I, :, :] = R

        # Divide by mass2
        Qlk[:, :, :] /= ctmqc_env['mass']

#        for I in ctmqc_env['QM_reps']

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



def calc_Qlk_Min17(runData):
    """
    Will calculate the quantum momentum as written in Min, 17.
    """
    ctmqc_env = runData.ctmqc_env
    if ctmqc_env['do_sigma_calc']:
        calc_Gossel_sigma(ctmqc_env)

    threshold = 0.995
    mask = [not any(i) for i in runData.ctmqc_env['adPops'] > threshold]
    reps_to_do = np.arange(ctmqc_env['nrep'])[mask]

    # Verified by hand for a 3 rep system
    ctmqc_env['WIJ'] = calc_WIJ(ctmqc_env)
    ctmqc_env['alpha'] = np.sum(ctmqc_env['WIJ'], axis=1)

    # Now calculate intercept
    Rlk = calc_Rlk(ctmqc_env)


    # Smooth out the intercept
    effR = get_effective_R(runData, Rlk)

    ctmqc_env['Rlk'] = Rlk
    ctmqc_env['effR'] = effR

    # Finally calculate Qlk
    Qlk = np.zeros((ctmqc_env['nrep'],
                    ctmqc_env['nstate'],
                    ctmqc_env['nstate']))
    if effR is False: return Qlk

    for I in reps_to_do:
        Qlk[I, :, :] = (ctmqc_env['alpha'][I] * ctmqc_env['pos'][I]) - effR
        for l in range(ctmqc_env['nstate']):
            Qlk[I, l, l] = 0.0

    # Only for 2 states
    if np.any(Qlk[:, 0, 1] != Qlk[:, 1, 0]):
        raise SystemExit("Qlk not symmetric!")

    Qlk /= ctmqc_env['mass']
#    print("Qlk = ", Qlk)
    return Qlk
'''

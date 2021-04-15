#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:11:43 2019

@author: oem
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


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
            pos = ctmqc_env['pos']
            RIv = pos[I]
            sig = ctmqc_env['sigma'] ** 2
            exponent[I, :] = -(RIv - pos)**2 / (2*sig)
#            for J in range(nRep):
#                RJv = ctmqc_env['pos'][J]
#                sJv = ctmqc_env['sigma'][J]
#
#                exponent[I, J] -= ( (RIv - RJv)**2 / (sJv**2))
#            print(exponent[I, :] - ()
#            exponent[I, :] = -np.sum((RIv - pos)**2/sig**2)
    else:
        for I in range(ctmqc_env['nrep']):
            pos = ctmqc_env['pos']
            RIv = pos[I]
            sig = ctmqc_env['sigma']
            exponent[I, :] = -(RIv - pos)**2 / (2*sig)

    return np.exp(exponent * 0.5) * prefact


def getChi(pos, sig):
    ctmqc_env = {'nrep': len(pos), 'pos': pos}
    ctmqc_env['sigma'] = sig
    chi = np.mean(calc_all_prod_gauss(ctmqc_env), axis=1)
    chi /= abs(integrate.trapz(chi, pos))
    return chi


pos = np.array(list(np.random.normal(2, 0.2, 200)) + list(np.random.normal(-5, 1.3, 200)))
minP, maxP, nBin = min(pos), max(pos), int(np.sqrt(len(pos)))
bins = np.linspace(minP, maxP, nBin+1)
binCounts = [len(pos[(pos >= m) & (pos <= M)])
             for m, M in zip(bins[:-1], bins[1:])]
binMap = {}
for count, (m, M) in enumerate(zip(bins[:-1], bins[1:])):
    for i in np.arange(len(pos))[(pos >= m) & (pos <= M)]:
        binMap[i] = count
revBinMap = {i : [] for i in set(binMap.values())}
for i in binMap: revBinMap[binMap[i]].append(i)
posBins = [np.mean([pos[j] for j in revBinMap[i] ])
           for i in set(revBinMap.keys())]
Hplot = np.array([binCounts[binMap[revBinMap[i][0]]] for i in revBinMap],
             dtype=float)
Hplot /= abs(integrate.simps(Hplot, posBins))

H = np.array([binCounts[binMap[i]] for i in range(len(pos))], dtype=float)
H /= abs(integrate.simps(H, pos))


def getInterpolatedH(pos, posBins, Hplot):
    """
    Will use linear interpolation between points on the Hamiltonian
    """
    posBins, Hplot = np.array(posBins), np.array(Hplot)

    if pos in posBins:
        return Hplot[posBins == pos][0]

    elif pos > max(posBins):
        x2 = max(posBins)
        x1 = max(posBins[posBins != x2])
        y2 = max(Hplot)
        y1 = max(Hplot[Hplot != y2])

    elif pos < min(posBins):
        x1 = min(posBins)
        x2 = min(posBins[posBins != x1])
        y1 = min(Hplot)
        y2 = min(Hplot[Hplot != y1])

    else:
        x2 = min(posBins[posBins > pos])
        x1 = max(posBins[posBins < pos])
        y2 = Hplot[posBins == x2][0]
        y1 = Hplot[posBins == x1][0]

    xdiff = x2 - x1
    ydiff = y2 - y1

    newH = y1 + (((pos - x1) / xdiff) * ydiff)
    return newH


# Get all the interpolated H values
allH = [getInterpolatedH(RI, posBins, Hplot) for RI in pos]


#chi = getChi(pos, 0.3)
#plt.plot(pos, allH, 'k.')
#plt.plot(pos, chi, 'b.')



dS = 0.01
ctmqc_env = {'nrep': len(pos), 'pos': pos}
allSig = np.arange(0.08, 2, dS)
allGrad = []
allP = []
for count, sigma in enumerate(allSig):
    grad = 0.0
    chi = getChi(pos, sigma)

    P = np.sum((allH - chi)**2)
    allP.append(P)

    if count % 1000 == 0:   print(sigma)


if np.min(allP) < 0.3:
    bestSig = allSig[np.argmin(allP)]

    print("Best Sigma = %.2g" % bestSig)
    print("Residuals = %.2g" % np.min(allP))

    chi=getChi(pos, bestSig)
    plt.plot(pos, chi, 'k.')
    plt.plot(pos, allH, 'r.')

else:
    print("Couldn't find the optimum sigma, residuals were too high.")

from __future__ import print_function

import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
from plottingResults import getData


def plot2(runData, func1, func2):
    f, a = plt.subplots(2)
    func1(runData, f=f, a=a[0])
    func2(runData, f=f, a=a[1])
    a[0].set_xlabel("")
    plt.show()


def get_ExtData(extData, model, mom):
    dfDeco = getattr(extData,
                         "mod%i_%sMom_deco" % (model, mom))
    dfPop = getattr(extData,
                        "mod%i_%sMom_pops" % (model, mom))
    mom = getattr(extData,
                      "mod%i_%sMom" % (model, mom))
    return dfPop, dfDeco, mom


def plotTotE(runData, f=False, a=False, params={}):
    """
    Will just plot the total energy against time on a single axis (good for
    plotting with other graphs).
    """
    lw = 0.5
    alpha = 0.2

    if a is False or f is False: f, a = plt.subplots()


    potE = np.sum(runData.allAdPop * runData.allE, axis=2)
    #kinE = 0.5 * runData.ctmqc_env['mass'] * (runData.allv**2)
    kinE = 0.5 * 2000. * (runData.allv**2)
    totE = potE + kinE

    a.plot(runData.allt, totE, lw=lw, alpha=alpha, **params)
    avgTotE = np.mean(totE, axis=1)
    a.plot(runData.allt, avgTotE, 'k', **params)

    # Now get the conservation level
    dt = runData.ctmqc_env['dt']
    fit = np.polyfit(runData.allt, avgTotE, 1)
    slope = fit[0] * dt * 41341.3745758
    annotateMsg = r"Energy Drift = %.2g" % (slope)
    annotateMsg += r" [$\frac{Ha}{atom ps}$]"
    ypos = avgTotE[100] / 2.
    xpos = runData.allt[100]
    a.annotate(annotateMsg, (xpos, ypos), fontsize=20)

    a.set_xlabel("Time [au]")
    a.set_ylabel("Tot. E [au]")
    print(r"Energy Drift = %.2g [Ha / (atom ps)]" % (slope))


def plotEcons(runData, f=False, a=False, params={}):
    """
    Will plot the conserved quantity along with the kinetice and potential
    energies.
    """
    lw = 0.5
    alpha = 0.2

    if a is False or f is False: f, a = plt.subplots(3)


    potE = np.sum(runData.allAdPop * runData.allE, axis=2)
    kinE = 0.5 * runData.ctmqc_env['mass'] * (runData.allv**2)
    totE = potE + kinE
    #a.plot(runData.allt, potE, 'g', lw=lw, alpha=alpha, **params)
    #a.plot(runData.allt, kinE, 'r', lw=lw, alpha=alpha, **params)
    a[0].plot(runData.allt, kinE[:, :], lw=lw, alpha=alpha)
    a[0].plot(runData.allt, np.mean(kinE, axis=1), 'r-', **params)
    a[1].plot(runData.allt, totE, lw=lw, alpha=alpha, **params)

    avgTotE = np.mean(totE, axis=1)
    #a.plot(runData.allt, np.mean(potE, axis=1), 'g', **params)
    #a.plot(runData.allt, np.mean(kinE, axis=1), 'r', **params)
    a[1].plot(runData.allt, avgTotE, 'k', **params)

    a[2].plot(runData.allt, potE[:, :], lw=lw, alpha=alpha)
    a[2].plot(runData.allt, np.mean(potE, axis=1), 'g-',
              **params)

    dt = runData.ctmqc_env['dt']
    fit = np.polyfit(runData.allt, avgTotE, 1)
    slope = fit[0] * dt * 41341.3745758
    annotateMsg = r"Energy Drift = %.2g" % (slope)
    annotateMsg += r" [$\frac{Ha}{atom ps}$]"
    ypos = avgTotE[100] / 2.
    xpos = runData.allt[100]
    a[1].annotate(annotateMsg, (xpos, ypos), fontsize=20)

    a[2].set_xlabel("Time [au]")
    a[0].set_ylabel("Kin. E [au]")
    a[1].set_ylabel("Tot. E [au]")
    a[2].set_ylabel(r"Pot. E [au]")
    a[0].set_title(annotateMsg)
    print(r"Energy Drift = %.2g [Ha / (atom ps)]" % (slope))


def plotClusters(runData, f=False, a=False, params={}):
    if a is False or f is False: f, a = plt.subplots()
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
              '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
              '#cab2d6', '#6a3d9a', '#ffff99', 'b', 'g',
              'r', 'c', 'm', 'k']

    for istep, stepCluster in enumerate(runData.allClusters):
	    for clustI in stepCluster:
	        inds = stepCluster[clustI]
	        pos = runData.allR[istep, inds]
	        time = np.ones(len(pos)) * runData.allt[istep]
	        a.plot(time, pos, '.', color=colors[clustI])



def Rabi(runData, f=False, a=False, params={}):
    H = runData.allH[0, 0]
    if a is False or f is False: f, a = plt.subplots()
    t = runData.allt
    delE = H[0, 0] - H[1, 1]
    Hab = H[0,1]
    sinPart = np.sin(0.5 * t * np.sqrt(delE**2 + 4*Hab**2))**2
    prefact = (4*Hab**2)/(delE**2 + 4 * Hab**2)
    rabiPops = 1 - (prefact * sinPart)
    return rabiPops


def plotRabi(runData, f=False, a=False, params={}):
    if a is False or f is False: f, a = plt.subplots()
    rabiPops = Rabi(runData)
    diPops = np.conjugate(runData.allu) * runData.allu
    a.plot(runData.allt, rabiPops, 'r', lw=3, alpha=0.5)
    a.plot(runData.allt, diPops[:, :, 0], 'k--', lw=1)
    a.set_ylabel("Diabatic Population")
    a.set_xlabel("Time [au]")
    a.set_title(r"Rabi Oscillation H$_{ab}$ = %.2g (Diab. Prop)" % runData.allH[0, 0, 0, 1])
    a.legend()
    plt.show()


def plotS26(runData, f=False, a=False, params={}):
    """
    Will plot the equation S26 that the Qlk should obey.
    """
    l, k = 0, 1
    S2612 = 2 * runData.allQlk[:, :, l, k] * (runData.allAdMom[:, :, k] - runData.allAdMom[:, :, l]) * runData.allAdPop[:, :, k] * runData.allAdPop[:, :, l]
    l, k = 1, 0
    S2621 = 2 * runData.allQlk[:, :, l, k] * (runData.allAdMom[:, :, k] - runData.allAdMom[:, :, l]) * runData.allAdPop[:, :, k] * runData.allAdPop[:, :, l]

    S2621 = np.sum(S2621, axis=1) # sum over reps
    S2612 = np.sum(S2612, axis=1) # sum over reps
    lw = 1
    alpha = 1

    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, S2612, lw=lw, alpha=alpha, **params)
    a.plot(runData.allt, S2621, lw=lw, alpha=alpha, **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel("S26")
    a.legend()


def plotAdMom(runData, f=False, a=False, params={}):
    """
    Will plot the adiabatic momentum
    """
    if a is False or f is False: f, a = plt.subplots()
    lw = 0.5
    alpha = 0.5
    avgf = np.mean(runData.allAdMom, axis=1)

    a.plot(runData.allt, runData.allAdMom[:, :, 0],
           color='r', lw=lw, alpha=alpha)
    a.plot(runData.allt, runData.allAdMom[:, :, 1],
           color='b', lw=lw, alpha=alpha)
    a.plot(runData.allt, avgf[:, 0], color='r')
    a.plot(runData.allt, avgf[:, 1], color='b')
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\mathbf{f}_{l}^{(I)}$ [au]")
    a.legend()


def plotAdEner(runData, f=False, a=False, params={}):
    """
    Will plot the adiabatic momentum
    """
    if a is False or f is False: f, a = plt.subplots()
    lw = 0.5
    alpha = 0.5
    avgf = np.mean(runData.allE, axis=1)

    a.plot(runData.allt, runData.allE[:, :, 0],
           color='r', lw=lw, alpha=alpha)
    a.plot(runData.allt, runData.allE[:, :, 1],
           color='b', lw=lw, alpha=alpha)
    a.plot(runData.allt, avgf[:, 0], color='r')
    a.plot(runData.allt, avgf[:, 1], color='b')
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$E_{l}^{(I)}$ [au]")
    a.legend()


def plotAdFrc(runData, f=False, a=False, params={}):
    """
    Will plot the adiabatic momentum
    """
    if a is False or f is False: f, a = plt.subplots()
    lw = 0.5
    alpha = 0.5
    avgf = np.mean(runData.allAdFrc, axis=1)

    a.plot(runData.allt, runData.allAdFrc[:, :, 0],
           color='r', lw=lw, alpha=alpha)
    a.plot(runData.allt, runData.allAdFrc[:, :, 1],
           color='b', lw=lw, alpha=alpha)
    a.plot(runData.allt, avgf[:, 0], color='r')
    a.plot(runData.allt, avgf[:, 1], color='b')
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\mathbf{Fad}_{l}^{(I)}$ [au]")
    a.legend()


def plotFrc(runData, f=False, a=False, params={}):
    """
    Will plot the total force
    """
    if a is False or f is False: f, a = plt.subplots()
    lw = 0.5
    alpha = 0.5
    avgf = np.mean(runData.allF, axis=1)

    a.plot(runData.allt, runData.allF,
           color='r', lw=lw, alpha=alpha)
    a.plot(runData.allt, avgf, color='r')
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\mathbf{F}_{l}^{(I)}$ [au]")
    a.legend()


def plotQMFrc(runData, f=False, a=False, params={}):
    """
    Will plot the total force
    """
    if a is False or f is False: f, a = plt.subplots()
    lw = 0.5
    alpha = 0.5
    avgf = np.mean(runData.allFqm, axis=1)

    a.plot(runData.allt, runData.allFqm,
           color='r', lw=lw, alpha=alpha)
    a.plot(runData.allt, avgf, color='r')
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\mathbf{Fqm}_{l}^{(I)}$ [au]")
    a.legend()


def plotRlk_Rl(runData, f=False, a=False, params={}):
    """
    Will plot the Rlk and Rl term on the same axis
    """
#    lw = 0.5
#    alpha = 0.5
    if a is False or f is False: f, a = plt.subplots()
#    if runData.allRl.shape[1] == 2:
#        a.plot(runData.allt, runData.allRl[:, 0], label=r"R$_{0}$", **params)
#        a.plot(runData.allt, runData.allRl[:, 1], label=r"R$_{1}$", **params)
#    else:
#        a.plot(runData.allt, runData.allRl[:, 0], 'k', lw=lw, alpha=alpha,
#               label=r"$R_{\nu, 0}^{(I)}$")
#        a.plot(runData.allt, runData.allRl[:, 1:], 'k', lw=lw, alpha=alpha)
    a.plot(runData.allt, runData.allRlk[:, 0, 1], 'r', label="Rlk", **params)
    a.plot(runData.allt, runData.allEffR[:, 0, 1, 0], 'g', label=r"R$_{eff}$", **params)
    a.plot(runData.allt, runData.allEffR[:, 0, 1, 1:], 'g', **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel("Intercept [au]")
    a.legend()


def plotRlk(runData, f=False, a=False, params={}):
    """
    Will plot the Rlk and Rl term on the same axis
    """
    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, runData.allRlk[:, 0, 1], 'k', label="Rlk", **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel("Intercept [au]")
    a.legend()


def plotDenom(runData, f=False, a=False, params={}):
    """
    Will plot the Rlk and Rl term on the same axis
    """
    if a is False or f is False: f, a = plt.subplots()
    C = runData.allAdPop
    f = runData.allAdMom
    denom = np.sum(C[:,:,0] * C[:,:,1] * (C[:,:,0] - C[:,:,1]), axis=1)
    a.plot(runData.allt, denom, 'k', **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel("Intercept Denominator [au]")
#    a.legend()


def plotRlk_gradRlk(runData, f=False, a=False, params={}):
    """
    Will plot the Rlk and Rl term on the same axis
    """
#    lw = 0.5
#    alpha = 0.5
    if a is False or f is False: f, ax = plt.subplots(2)
    a = ax[0]

#    if runData.allRl.shape[1] == 2:
#        a.plot(runData.allt, runData.allRl[:, 0], label=r"R$_{0}$", **params)
#        a.plot(runData.allt, runData.allRl[:, 1], label=r"R$_{1}$", **params)
#    else:
#        a.plot(runData.allt, runData.allRl[:, 0], 'k', lw=lw, alpha=alpha,
#               label=r"$R_{\nu, 0}^{(I)}$")
#        a.plot(runData.allt, runData.allRl[:, 1:], 'k', lw=lw, alpha=alpha, **params)

    a.plot(runData.allt, runData.allRlk[:, 0, 1], 'r', label="Rlk", **params)
    a.plot(runData.allt, runData.allEffR[:, 0, 1], 'g', label=r"R$_{eff}$", **params)

    a.set_ylabel("Intercept [au]")
    a.legend()

    a = ax[1]
    a.plot(runData.allt,
           g)
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\frac{\delta R_{lk, \nu}^{0}}{\delta t}$ [au]")


def plotQlk(runData, f=False, a=False, params={}):
    lw = 0.25
    alpha = 0.5

    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, runData.allQlk[:, :, 0, 1], 'k', lw=lw, alpha=alpha, **params)
    #a.plot(runData.allt, runData.allQlk[:, :, 0, 0], 'g', lw=lw, alpha=alpha, **params)
    #a.plot(runData.allt, runData.allQlk[:, :, 1, 1], 'r', lw=lw, alpha=alpha, **params)
    #a.plot(runData.allt, runData.allQlk[:, :, 1, 0], 'b', lw=lw, alpha=alpha, **params)


def plotAlpha(runData, f=False, a=False, params={}):
   lw = 0.25
   alpha = 0.5

   if a is False or f is False: f, a = plt.subplots()
   a.plot(runData.allt, runData.allAlpha, lw=lw, alpha=alpha, **params)


def plotPops(runData, f=False, a=False, params={}):
    lw = 0.25
    alpha = 0.5

    #gossData = getData.GosselData()
    #model = runData.ctmqc_env['tullyModel']
    #mom = int(runData.ctmqc_env['velInit'] / 0.0005)
    #if mom > 20: mom = 'high'
    #else: mom = 'low'
    #dfPops, _, _ = get_ExtData(gossData, model, mom)

    if a is False or f is False: f, a = plt.subplots()
    #a.plot(dfPops['CTMQC_x'], dfPops['CTMQC_y'], 'r', **params)
    a.plot(runData.allt, runData.allAdPop[:, :, 1], 'b', lw=lw, alpha=alpha, **params)
    a.plot(runData.allt, runData.allAdPop[:, :, 0], 'r', lw=lw, alpha=alpha, **params)
    a.plot(runData.allt, np.mean(runData.allAdPop[:, :, 0], axis=1), 'r',
           **params)
    a.plot(runData.allt, np.mean(runData.allAdPop[:, :, 1], axis=1), 'b',
            **params)
    a.set_ylabel("Adiab. Pop.")
    a.set_xlabel("Time [au]")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])
    a.legend()


def plotDiPops(runData, f=False, a=False, params={}):
    lw = 0.25
    alpha = 0.5
    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, runData.allu[:, :, 1], 'b', lw=lw, alpha=alpha, **params)
    a.plot(runData.allt, runData.allu[:, :, 0], 'r', lw=lw, alpha=alpha, **params)
    a.plot(runData.allt, np.mean(runData.allu[:, :, 0], axis=1), 'r',
           **params)
    a.plot(runData.allt, np.mean(runData.allu[:, :, 1], axis=1), 'b',
           **params)
    a.set_ylabel("Diab. Pop.")
    a.set_xlabel("Time [au]")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])
    a.legend()


def plotNorm(runData, f=False, a=False, params={}):
    lw = 0.25
    alpha = 0.5
    dt = runData.ctmqc_env['dt']
    # Sum over states
    allNorms = np.sum(runData.allAdPop, axis=2)
    # Mean over all reps
    avgNorms = np.mean(allNorms, axis=1)
    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, allNorms, 'r', lw=lw, alpha=alpha, **params)
    a.plot(runData.allt, avgNorms, 'r', **params)
    a.set_ylabel("Norm")
    a.set_xlabel("Time [au]")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])

    fit = np.polyfit(runData.allt, avgNorms, 1)
    print("Norm Drift = %.2g [$ps^{-1}$]" % (fit[0] * 41341.3745758 * dt))


def plotDeco(runData, f=False, a=False, params={}):
    if a is False or f is False: f, a = plt.subplots()

    lw = 0.1
    alpha = 0.5
    deco = runData.allAdPop[:, :, 0] * runData.allAdPop[:, :, 1]
    #gossData = getData.GosselData()
    #model = runData.ctmqc_env['tullyModel']
    #mom = int(runData.ctmqc_env['velInit'] / 0.0005)
    #if mom > 20: mom = 'high'
    #else: mom = 'low'
    #_, dfDeco, _ = get_ExtData(gossData, model, mom)

    a.plot(runData.allt, deco, 'k', lw=lw, alpha=alpha, **params)
    #a.plot(dfDeco['CTMQC_x'], dfDeco['CTMQC_y'], 'r', **params)
    a.plot(runData.allt, np.mean(deco, axis=1), 'k', **params)
    a.set_ylabel("Coherence")
    a.set_xlabel("Time [au]")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])


def plotSigma(runData, f=False, a=False, params={}):
    """
    Will plot sigmal against time
    """
    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, runData.allSigma[:, :], 'k',
           lw=0.5, alpha=0.5)
    a.plot(runData.allt, np.mean(runData.allSigma, axis=1),
           'k', **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\sigma^{(I)}$ [au]")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])


def plotSigmal(runData, f=False, a=False, params={}):
    """
    Will plot sigmal against time
    """
    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, runData.allSigmal[:, 0], 'r', **params)
    a.plot(runData.allt, runData.allSigmal[:, 1], 'b', **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\sigma_{l}$")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])
    a.legend()


def plotPos(runData, f=False, a=False, params={}):
    """
    Will plot sigmal against time
    """
    lw = 0.3
    alpha = 0.7

    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, runData.allR, lw=lw, alpha=alpha, **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\mathbf{R}^{(I)}$")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])


def plotVel(runData, f=False, a=False, params={}):
    """
    Will plot sigmal against time
    """
    lw = 0.3
    alpha = 0.7

    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, runData.allv, lw=lw, alpha=alpha, **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\mathbf{v}^{(I)}$")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])


def plotNACV(runData, f=False, a=False, params={}):
    """
    Will plot NACV against position
    """
    lw = 1
    alpha = 0.7

    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, runData.allNACV[:, :, 0, 1], lw=lw, alpha=alpha, **params)
    #a.plot(runData.allt, runData.allNACV[:, :, 0, 0], lw=lw, alpha=alpha, **params)
    #a.plot(runData.allt, runData.allNACV[:, :, 1, 1], lw=lw, alpha=alpha, **params)
    #a.plot(runData.allt, runData.allNACV[:, :, 1, 0], lw=lw, alpha=alpha, **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"NACV")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])


def plotH(runData, f=False, a=False, params={}):
    """
    Will plot NACV against position
    """
    lw = 1
    alpha = 0.7

    if a is False or f is False: f, a = plt.subplots()
    a.plot(runData.allt, runData.allH[:, :, 0, 1], lw=lw, alpha=alpha, **params)
    a.plot(runData.allt, runData.allH[:, :, 0, 0], lw=lw, alpha=alpha, **params)
    a.plot(runData.allt, runData.allH[:, :, 1, 1], lw=lw, alpha=alpha, **params)
    a.plot(runData.allt, runData.allH[:, :, 1, 0], lw=lw, alpha=alpha, **params)
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"H [au]")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])


def plot_single_Epot_frame(data, istep, saveFolder=False, f=False, a=False, params={}):
    """
    Will plot a single frame
    """
    potE = np.sum(data['|C|^2'][istep] * data['E'][istep], axis=1)
    if a is False or f is False: f, a = plt.subplots()
    a.plot(data['x'], data['E'][:, :, 0], 'r', **params)
    a.plot(data['x'], data['E'][:, :, 1], 'b', **params)
    a.plot(data['x'][istep], potE, 'k', **params)
    a.set_ylabel("Energy [Ha]")
    a.set_xlabel("Nucl. Pos [bohr]")
    if isinstance(saveFolder, str):
        f.savefig("%s/%06i.png" % (saveFolder, istep))
        plt.close()
    else:
        plt.show()

def plot_Epot_wrapper_func(args):
    """
    Acts as a wrapper to use the multiprocessing module.
    """
    if len(args) > 3:
        raise SystemExit("More arguments than is possible for the "
                         + "plot_single_Epot_frame function." +
                         "\n\t* Num Args Given: %i" % len(args) +
                         "\n\t* Max Args Allowed: 3")


    plot_single_Epot_frame(*args)


def orderFiles(folder, ext, badStr=False, replacer=False, params={}):
    """
    Will rename files to reorder them for ffmpeg.
    """
    # First find the files and sort them
    allFiles = [f for f in os.listdir(folder) if ext in f]
    if badStr is not False:
        allFiles = [f for f in allFiles if badStr not in f]
    if replacer is not False:
        allFiles = [f.replace(replacer, "") for f in allFiles]
    allNums = [int(f.replace(ext, "")) for f in allFiles]
    allFiles = [i[1] for i in sorted(zip(allNums, allFiles))]

    # Now change the names
    maxZeros = len(str(len(allFiles)))
    for i, oldF in enumerate(allFiles):
        newF = "0" * (maxZeros - len(str(i))) + str(i) + ".png"
        os.rename("%s/%s" % (folder, oldF), "%s/%s" % (folder, newF))

    ffmpegCmd = "ffmpeg -framerate 120 "
    ffmpegCmd += "-i %s/%%0%id.png potE_ani.mp4" % (folder, maxZeros)
    print("Use the following command to stitch the pics together:")
    print("\t`%s`" % ffmpegCmd)


def plotEpotTime(runData, which_steps=False, saveFolder=False, step=1):
    """
    Will plot the potential energy over time for either each step or just the
    one specified.
    """
    if which_steps is False:
        which_steps = range(0, runData.ctmqc_env['iter'], step)
    if isinstance(which_steps, int):
        data = {'x': runData.allR, 'E': runData.allE,
                '|C|^2': runData.allAdPop}
        print("\n\n\n")
        print(which_steps)
        print(isinstance(which_steps, int))
        print("\n\n\n")
        plot_single_Epot_frame(data, which_steps, saveFolder)
    else:
        if not isinstance(which_steps, (type(np.array(1)), list)):
            raise SystemExit("Sorry the argument `which_steps` has been"+
                             " inputted incorrectly. It should be an int, " +
                             "a list or a numpy array.")
        pool = multiprocessing.Pool()

        # First create the arguments to feed into the wrapper function
        data = {'x': runData.allR, 'E': runData.allE,
                '|C|^2': runData.allAdPop}
        data = [data] * len(which_steps)
        saveFolders = [saveFolder] * len(which_steps)

        # Feed in the arguments zipped up
        pool.map(plot_Epot_wrapper_func,
                            zip(data, which_steps, saveFolders))

        orderFiles(saveFolder, ".png")


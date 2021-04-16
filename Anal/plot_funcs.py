import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import pandas as pd

import plot_utils as pu

show = True

def get_fig_ax(fig_ax):
    """
    Will return a figure and axis if not already created
    """
    if fig_ax is False:  return plt.subplots()
    else:           return f, a


def plot_all_ener(folder, fig_ax=False):
    """
    Will plot the energy levels of the simulations
    """
    tFP = f"{folder}/time.npy"
    eFP = f"{folder}/E.npy"

    time = np.load(tFP) * pu.tConv
    E = np.load(eFP)

    f, a = get_fig_ax(fig_ax)

    a.plot(time, E[:, :, 0], 'r', lw=0.7, alpha=0.1)
    a.plot(time, np.mean(E[:, :, 0], axis=1), 'r--')
    a.plot(time, E[:, :, 1], 'b', lw=0.7, alpha=0.1)
    a.plot(time, np.mean(E[:, :, 1], axis=1), 'b--')

    if show: plt.show()

    return f, a


def groupbyMean(df):
    if len(df) > 70:
        return df.mean()

def avg_pos_pops(R, pops, res=0.3):
    """
    Will bin the pops and pos and then average by bin.
    """
    nrep = R.shape[1]
    allDf = [pd.DataFrame({'R': R[:, i], 'ground': pops[:, i, 0], 'excited': pops[:, i, 1]}) for i in range(nrep)]
    df = pd.concat(allDf)
    df['R'] = (df['R'] // res) * res
    avgDf = df.groupby('R').apply(groupbyMean)
    avgDf = avgDf.dropna()#.rolling(window=10).mean()

    R = avgDf.index.to_numpy()
    pops = np.swapaxes(np.array([avgDf['ground'].to_numpy(), avgDf['excited'].to_numpy()]), 0, 1)
    return R, pops

def plot_all_pops(folder, fig_ax=False):
    """
    Will plot all the populations with an average in a folder.
    """
    tFP = f"{folder}/time.npy"
    RFP = f"{folder}/pos.npy"
    pFP = f"{folder}/|C|^2.npy"
    EFP = f"{folder}/E.npy"
    dlkFP = f"{folder}/NACV.npy"

    t = np.load(tFP) * pu.tConv
    R = np.load(RFP)
    pops = np.load(pFP)
    E = np.load(EFP)
    dlk = np.load(dlkFP)

    #f, a = get_fig_ax(fig_ax)
    f, axes = plt.subplots(2, 1)

    a = axes[0]
    Ravg, popsavg = avg_pos_pops(R, pops)
    a.plot(R, pops[:, :, 0], 'r', lw=0.7, alpha=0.1)
    a.plot(Ravg, popsavg[:, 0], 'r--', label="Ground")
    #a.plot(t, np.mean(pops[:,:,0]*pops[:,:,1], axis=1), 'r--', label="Ground") #popsavg[:, 0], 'r--', label="Ground")
    #a.plot(t, np.mean(pops, axis=1)[:,0], 'r--', label="Ground") #popsavg[:, 0], 'r--', label="Ground")

    #a.plot(time, pops[:, :, 1], 'b', lw=0.7, alpha=0.1)
    #a.plot(time, np.mean(pops[:, :, 1], axis=1), 'b--',
    #      label="Excited")

    a.set_ylabel(r"$|C_l^{(I)}|^2$")

    a1 = axes[1]
    a1tw = a1.twinx()
    avgE = np.mean(E, axis=1)
    #a1.plot(np.mean(R, axis=1), avgE[:, 0], 'r', lw=1.5)
    a1.plot(R, E[:,:,0], 'r', lw=0.7, alpha=0.2)
    a1.plot(R, E[:,:,1], 'b', lw=0.7, alpha=0.2)
    a1tw.plot(R, dlk[:, :, 0, 1], 'k--', lw=0.7)
    a1.set_xlabel("R [bohr]")
    a1.set_ylabel("E [hartree]")
    a1tw.set_ylabel(f"NACV [bohr$^{-1}$]")
    #a.legend(loc="best", fontsize=18)


    xlim = a.get_xlim()
    a1.set_xlim(xlim)
    #a1tw.set_ylim([-7, 8])

    if show: plt.show()

    return f, a


def plot_all_norm(folder, fig_ax=False):
    """
    Will plot the norms for each traj and take average and get line of best fit
    """
    tFP = f"{folder}/time.npy"
    pFP = f"{folder}/|C|^2.npy"

    time = np.load(tFP) * pu.tConv
    pops = np.load(pFP)
    norms = pops[:,:,0] + pops[:, :, 1]
    avgNorm = np.mean(norms, axis=1)

    f, a = get_fig_ax(fig_ax)

    a.plot(time, norms, 'k-', lw=0.7, alpha=0.1)
    a.plot(time, avgNorm, 'k--',
             label="Average Norm")

    fit = np.polyfit(time, norms, 1)
    avgFit = np.mean(fit, axis=1)
    fit, fitErr = curve_fit(lambda x, m, c: (m*x) + c,
                            time, avgNorm, p0=avgFit)
    fitErr = np.sqrt(np.diag(fitErr))

    xlims, ylims = a.get_xlim(), a.get_ylim()
    a.annotate(f"Norm Drift: {fit[0]:.1g} " + r"$\pm$ " + f"{fitErr[0]:.1g}",
               (xlims[0] + 0.8*(xlims[1]-xlims[0]),
                ylims[0] + 0.8*(ylims[1]-ylims[0])),
               fontsize=18)

    if show: plt.show()

    return f, a


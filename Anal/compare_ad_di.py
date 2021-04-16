import plot_funcs as pf
import plot_utils as pu

import matplotlib.pyplot as plt
import os
import numpy as np



adFolder = "../adiabProp"
diFolder = "../diabProp"


pops_or_coh = "coherences"
CTMQC_or_Ehren = "CTMQC"
Mom_High_or_Low = "high"
do_traj = True
do_sig = False

f, a = plt.subplots(2, 2, figsize=(16,9))


def plot_data(folder, a, hc='b', lc='r'):
    splitter = folder.split('/')
    if 'Kinit' not in splitter[-1]: return False

    mom = splitter[-1].split('_')[-1]
    model = splitter[-2].split('_')[-1]

    isCT = 'CTMQC' in splitter[-3]
    isHigh = int(mom) > 20

    do_CT = CTMQC_or_Ehren == "CTMQC"
    do_high = Mom_High_or_Low == "high"
    if isHigh:
        if not do_high: return False
    else:
        if do_high: return False
    if isCT:
        if not do_CT: return False
    else:
        if do_CT: return False

    mod = int(model) - 1
    i = mod // 2
    j = mod - (i*2)

    t = np.load(f"{folder}/time.npy")
    pops = np.load(f"{folder}/|C|^2.npy")

    if pops_or_coh == "coherences":
        pops = np.reshape(pops[:, :, 0] * pops[:, :, 1], (len(pops), len(pops[0]), 1))

    if do_traj:
        #a[i, j].plot(t, pops[:, :, 1], lw=0.7, color=hc, alpha=0.2)
        a[i, j].plot(t, pops[:, :, 0], lw=0.7, color=lc, alpha=0.2)

    if do_sig:
        sigma =  np.load(f"{folder}/sigma.npy")
        a[i,j].plot(t, sigma, color='k')

    ln, = a[i, j].plot(t, np.mean(pops[:,:,0], axis=1), color=lc, alpha=1, ls='--', lw=3)

    return (mom, model, isCT, isHigh, i, j, ln)


for folder, folders, files in os.walk(adFolder):
    cont = plot_data(folder, a)
    if cont is False: continue
    mom, model, isCT, isHigh, i, j, ln = cont
    ln.set_label("Adiabatic Propagator")

for folder, folders, files in os.walk(diFolder):
    cont = plot_data(folder, a, 'y', 'g')
    if cont is False: continue
    mom, model, isCT, isHigh, i, j, ln = cont
    a[i,j].set_title(f"Model {model}")
    ln.set_label("Diabatic Propagator")

if not do_traj:
    plt.legend(fontsize=24)

trajStr = "_noTraj" if do_traj else "_wTraj"
plt.savefig(f"/home/matt/Documents/PhD/PhD_Thesis/img/CTMQC/TullyModels/CTMQC_ad_vs_di{trajStr}_{pops_or_coh}.png", dpi=300)
plt.show()

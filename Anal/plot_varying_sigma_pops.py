GosselDataFolder="/home/matt/Data/Work/CTMQC/GosselData"
FredericaDataFolder="/home/matt/Data/Work/CTMQC/FredericaData"
MyDataFolder="/home/matt/Documents/PhD/Code/1D_tully_CTMQC/VaryingSigma_2nd"

import plot_utils

import matplotlib.pyplot as plt
import numpy as np
import os


colors = {0.1: 'r', 0.2: 'b', 0.3: 'y', 0.35: 'o'}
Ehren = False #"Ehren" in MyDataFolder
mom = 'high'
plotCount = 0
for pops_or_coh in ('pops', 'coherence'):
    if pops_or_coh == 'pops':
        fig_ax = plot_utils.create_fig_ax(r"$\frac{1}{N_{traj}} \sum_{I} |C^{(I)}_{0}|^2 $")
    elif pops_or_coh == "coherence":
        fig_ax = plot_utils.create_fig_ax(r"coherence")
    else:
        raise SystemExit("Please set 'pops_or_coh' to either 'coherence' or 'pops'")


    gossData = plot_utils.read_data(GosselDataFolder)
    fredData = plot_utils.read_data(FredericaDataFolder)
    myData = plot_utils.read_data(MyDataFolder, plot_utils._read_numpy)

    allMom = plot_utils.plot_FredGoss_data(gossData,
                                           False,
                                           fig_ax,
                                           pops_or_coh,
                                           mom,
                                           plot_vals=('Exact',))

    f, axes = fig_ax
    data = myData[pops_or_coh]
    lines = {}
    for sig in data:
        sigma = sig.replace("Sigma_", "")
        for mod in data[sig]:
            model = int(mod.replace("Model", "")) - 1
            i = model // 2
            j = model % 2
            for momStr in data[sig][mod]:
                t, pops = data[sig][mod][momStr]
                if pops_or_coh == "coherence":
                    y = np.mean(pops, axis=1)
                else:
                    y = np.mean(pops, axis=1)[:, 0]
                ln, = axes[i, j].plot(t * plot_utils.tConv,  y,
                                      label=r"$\sigma = %.2f$" % float(sigma))
                lines.setdefault(sigma, ln)

    if plotCount == 1:
        axes[1, 0].legend(fontsize=22, loc='upper left')
        xlim = axes[1,0].get_xlim()
        xrange_ = xlim[1] - xlim[0]
        axes[1, 0].set_xlim([-xrange_/10, xlim[1]])

    plotCount += 1


    plt.savefig(f"/home/matt/Documents/PhD/PhD_Thesis/img/CTMQC/TullyModels/CTMQC_varyingSigma_{mom}Mom_{pops_or_coh}.png", dpi=260)

os.chdir("/home/matt/Documents/PhD/PhD_Thesis/img/CTMQC/TullyModels/")
#os.system(f"convert +append {typeStr}_lowMom_coherence.png {typeStr}_highMom_coherence_1.png tmp.png")
#os.system(f"convert +append {typeStr}_lowMom_pops.png {typeStr}_highMom_pops_1.png tmp2.png")
os.system(f"convert -append CTMQC_varyingSigma_{mom}Mom_pops.png CTMQC_varyingSigma_{mom}Mom_coherence.png tmp_CTMQC_VarSig_{mom}Mom.png")

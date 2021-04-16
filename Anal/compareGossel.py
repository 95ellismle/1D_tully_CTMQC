GosselDataFolder="/home/matt/Data/Work/CTMQC/GosselData"
FredericaDataFolder="/home/matt/Data/Work/CTMQC/FredericaData"
MyDataFolder="/home/matt/Documents/PhD/Code/1D_tully_CTMQC/AdiabPropFinal/Ehren"

import plot_utils

import matplotlib.pyplot as plt
import os

Ehren = "Ehren" in MyDataFolder
mom = 'high'
pops_or_coh = "pops"
for mom in ('high', 'low'):
    print(mom)
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

        if Ehren:
            allMom = plot_utils.plot_FredGoss_data(gossData,
                                                   fredData,
                                                   fig_ax,
                                                   pops_or_coh,
                                                   mom,
                                                   ['Ehren'])
        else:
            allMom = plot_utils.plot_FredGoss_data(gossData,
                                                   fredData,
                                                   fig_ax,
                                                   pops_or_coh,
                                                   mom)

        print(allMom)
        plot_utils.plot_my_data(myData, fig_ax, allMom, pops_or_coh)



        if Ehren:
            plt.savefig(f"/home/matt/Documents/PhD/PhD_Thesis/img/CTMQC/TullyModels/Ehren_{mom}Mom_{pops_or_coh}_1.png", dpi=240)
        else:
            plt.savefig(f"/home/matt/Documents/PhD/PhD_Thesis/img/CTMQC/TullyModels/CTMQC_{mom}Mom_{pops_or_coh}_1.png", dpi=240)
        #plt.show()
        plt.close()

os.chdir("/home/matt/Documents/PhD/PhD_Thesis/img/CTMQC/TullyModels/")
if Ehren:  typeStr = "Ehren"
else: typeStr = "CTMQC"

os.system(f"convert +append {typeStr}_lowMom_coherence.png {typeStr}_highMom_coherence_1.png tmp.png")
os.system(f"convert +append {typeStr}_lowMom_pops.png {typeStr}_highMom_pops_1.png tmp2.png")
os.system(f"convert -append tmp.png tmp2.png {typeStr}_LitComp_1.png")

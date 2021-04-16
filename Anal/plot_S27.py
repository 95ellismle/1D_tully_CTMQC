import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import os

f, a = plt.subplots(figsize=(16, 9))

fontsize=30



root_folder = "/home/matt/Documents/PhD/Code/1D_tully_CTMQC/FullGossel_std/CTMQC/"

def calcS27(Qlk, fl, Cl):
    """
    Will calculate the S27 conservation equation.
    """
    return np.sum(Qlk[:, :, 0, 1] * (fl[:, :, 0] - fl[:, :, 1]) * Cl[:
, :, 0] * Cl[:, :, 1], axis=1)


for model_num in range(1, 5):
    model_fold = f"Model_{model_num}"
    fold = f"{root_folder}/{model_fold}"
    model_name = model_fold.replace("_", " ").title()

    mom_folds = os.listdir(fold)
    max_mom = max([int(re.findall("[0-9]+", i)[0]) for i in mom_folds])
    folder = f"{fold}/Kinit_{max_mom}"

    t = np.load(f"{folder}/time.npy")
    t /= 41.34 # Convert to fs
    fl = np.load(f"{folder}/f.npy")
    Cl = np.load(f"{folder}/|C|^2.npy")
    Qlk = np.load(f"{folder}/Qlk.npy")


    a.plot(t, calcS27(Qlk, fl, Cl), lw=0.7, alpha=0.5, label=model_name)



a.set_xlabel("Time [fs]", fontsize=fontsize)
a.set_ylabel(r"$\sum_{I} \mathcal{Q}_{lk, \nu}^{(I)}(t) \left( f_{k, \nu}^{(I)} - f_{l, \nu}^{(I)} \right) |C_{k}^{(I)} (t)|^2 |C_{l}^{(I)} (t)|^2$", fontsize=fontsize)
plt.xticks(fontsize=fontsize*0.8)
plt.yticks(fontsize=fontsize*0.8)

a.legend(fontsize=fontsize)
#plt.tight_layout()
plt.savefig("/home/matt/Documents/PhD/PhD_Thesis/img/CTMQC/TullyModels/CTMQC_S27.png", dpi=300)
#plt.show()


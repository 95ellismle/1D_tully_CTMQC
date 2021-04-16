import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json


def lin(x, m, c): return x*m + c


folder = "/home/matt/Documents/PhD/Code/1D_tully_CTMQC/FullGossel_std"

norms = []
dt = []

all_files = os.walk(folder)
for currFold, folds, files in all_files:
    if '|C|^2.npy' in files:
        fp = "%s/|C|^2.npy" % currFold
        fpT = "%s/time.npy" % currFold
        pops = np.load(fp)
        t = np.load(fpT)
        #info = np.load(fpI, allow_pickle=True)

        norm = np.mean(pops[:, :, 0] + pops[:, :, 1], axis=1)

        fit = np.polyfit(t, norm, 1)
        popt, pcov = curve_fit(lin, t, norm, p0=fit)


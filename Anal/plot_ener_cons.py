import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import integrate as INT


folder = "../FullGossel_std/CTMQC"


def calcEhEner(E, C):
    return np.sum(E*C, axis=2)


def calcTerm2(C, t):
    diffT = t[1] - t[0]
    dC_dt = np.gradient(C, diffT, axis=0)
    ABst = np.sum(dC_dt * np.conjugate(C), axis=2)
    return ABst.imag * complex(0, 1)


def calcTerm3(C, v, dlk):
    Cst = np.conjugate(C)
    term = Cst[:, :, 0] * C[:, :, 1] * v * dlk[:, :, 0, 1] + Cst[:, :, 1] * C[:, :, 0] * v * dlk[:, :, 1, 0]
    return term


def calcTerm4(Cpops, v, f):
    term = Cpops[:, :, 0] * v * f[:, :, 0] + Cpops[:, :, 1] * v * f[:, :, 1]
    return term



fig, axes = plt.subplots(2, 2)

i, j = 0, 0
for mod_num in range(1, 5):
    mod_fold  = f"{folder}/Model_{mod_num}"

    for mom_fold in os.listdir(mod_fold):
        fold = f"{mod_fold}/{mom_fold}"

        t = np.load(f"{fold}/time.npy")
        E = np.load(f"{fold}/E.npy")
        C = np.load(f"{fold}/|C|^2.npy")
        coeffC = np.load(f"{fold}/C.npy")
        v = np.load(f"{fold}/vel.npy")
        f = np.load(f"{fold}/f.npy")
        Feh = np.load(f"{fold}/Feh.npy")
        Fqm = np.load(f"{fold}/Fqm.npy")
        R = np.load(f"{fold}/pos.npy")
        dlk = np.load(f"{fold}/NACV.npy")

        Ftot = Feh + Fqm

        potE1 = calcEhEner(C, E)
        potE2 = calcTerm2(coeffC, t)
        potE3 = calcTerm3(coeffC, v, dlk) * complex(0, 1)
        potE4 = calcTerm4(C, v, f)
        potE5 = calcTerm3(coeffC, v, dlk).imag

        potE = (potE1 + potE2 + potE3 + potE4 + potE5).real
        Fest = -np.diff(potE[:, 0]) / np.diff(R[:, 0])
        #Fest = -np.gradient(potE[:, 0], np.diff(R[:, 0])[4000])

        #potEest

        kinE = v**2 * 0.5 * 2000
        Etot = np.mean(potE + kinE, axis=1)


        #axes[i, j].plot(t, potE)
        axes[i, j].plot(t[1:], -np.mean(np.diff(potE1, axis=0)/np.diff(R, axis=0), axis=1))
        #axes[i, j].plot(t, potE1, lw=0.7, color='r')
        #axes[i, j].plot(t, potE2, lw=0.7, color='b')
        #axes[i, j].plot(t, potE3, lw=0.7, color='g')
        #axes[i, j].plot(t, potE4, lw=0.7, color=(0.7, 0.7, 0))
        #axes[i, j].plot(t, potE5, lw=0.7, color=(0, 0.7, 0.7))

        print(np.polyfit(t, Etot, 1) * 1000 * 41.23)
        #axes[i, j].plot(t[:len(Fest)], Fest)
        #axes[i, j].plot(t, Ftot[:, 0])
        break

    i += 1
    if (mod_num % 2 == 0): j += 1
    i %= 2

plt.show()

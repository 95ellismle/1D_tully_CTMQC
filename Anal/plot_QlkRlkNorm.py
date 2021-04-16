import matplotlib.pyplot as plt
import os
import numpy as np


folder = "../RlkSpike/Model_1/Kinit_15"

t = np.load(f"{folder}/time.npy")
Rlk = np.load(f"{folder}/Rlk.npy")
pops = np.load(f"{folder}/|C|^2.npy")
Qlk = np.load(f"{folder}/Qlk.npy")
f = np.load(f"{folder}/f.npy")

norm = pops[:, :, 0] + pops[:, :, 1]
t /= 41.34
denom = np.sum(pops[:,:,0] * pops[:,:,1] * (f[:,:,0] - f[:,:,1]), axis=1)

f, a = plt.subplots(4, figsize=(20, 10))

# Plot Norm
a[0].plot(t, norm, lw=0.7, alpha=0.2, color='r')
a[0].plot(t, np.mean(norm, axis=1), lw=2,
          ls='--', color=(0.7, 0, 0))
a[0].set_ylabel(r"$\sum_{k}^{N_{st}}|C_{k}^{(I)}(t)|^2$")

# Plot Qlk
a[1].plot(t, Qlk[:, :, 0, 1], lw=0.7, alpha=0.4,
          color='k')
a[1].set_ylabel(r"$Q_{lk, \nu}^{(I)} (t)$")

# Plot Rlk
a[2].plot(t, Rlk[:, 0, 1])
a[2].set_ylabel(r"$R_{lk, \nu}$")
#a[2].set_ylim([-200, 200])

# Plot Denominator
a[3].plot(t, denom)
a[3].set_ylabel(r"$R_{lk, \nu}$"+"\nDenominator")
a[3].axhline(0, ls='--', lw=0.7, color='k')


# Draw vertical line on axes for zero crossing
firstZeroEscape = np.arange(len(denom))[np.abs(denom - 0) > 1][0]
newT = t[firstZeroEscape:]
newDenom = denom[firstZeroEscape:]
zeroCrossInd = np.argmin(np.abs(newDenom - 0))
zeroCrossT = newT[zeroCrossInd]

# Make pretty axes
for i in range(len(a)-1):
    a[i].set_xticks([])
    a[i].spines['bottom'].set_visible(False)

for ax in a: ax.axvline(zeroCrossT, color='k', lw=10, alpha=0.1)
a[-1].set_xlabel("Time [fs]")
plt.tight_layout()

plt.savefig("/home/matt/Documents/PhD/PhD_Thesis/img/CTMQC/TullyModels/Spikes/RlkDenom_Rlk_Qlk_Norm.png", dpi=220)
plt.show()

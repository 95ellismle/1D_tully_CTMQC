import matplotlib.pyplot as plt
import numpy as np


folder = "../FullGossel_Rlk_100Reps/CTMQC/Model_4/Kinit_40"

t = np.load(f"{folder}/time.npy")
f = np.load(f"{folder}/f.npy")


fig, a = plt.subplots(2)
a[0].plot(t, f[:, :, 0] - f[:, :, 1], lw=0.7, color='k', alpha=0.2)


plt.show()

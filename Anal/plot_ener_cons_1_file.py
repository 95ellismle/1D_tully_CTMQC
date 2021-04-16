import matplotlib.pyplot as plt
import numpy as np


fold = '/home/matt/Documents/PhD/Code/1D_tully_CTMQC/Model3Test/CTMQC/Model_3/Kinit_30'

t =    np.load(f"{fold}/time.npy")
dlk =  np.load(f"{fold}/NACV.npy")
pops = np.load(f"{fold}/|C|^2.npy")
C =    np.load(f"{fold}/C.npy")
E =    np.load(f"{fold}/E.npy")
f =    np.load(f"{fold}/f.npy")
Fqm =  np.load(f"{fold}/Fqm.npy")
Feh =  np.load(f"{fold}/Feh.npy")
v =    np.load(f"{fold}/vel.npy")
R =    np.load(f"{fold}/pos.npy")
m =    2000


def EhrenPot(pops, E):
    return np.sum(E * pops, axis=2)


def CalcPotTerm2(C, t):
    Cst = np.conjugate(C)
    dt = t[1] - t[0]
    dC_dt = np.gradient(C, dt, axis=0)
    term = np.sum(Cst * dC_dt, axis=2)
    term[-1] = term[-2] + dt*(term[-2] - term[-3])
    return term


def CalcPotTerm3(C, v, dlk):
    Cst = np.conjugate(C)
    term = Cst[:,:,0]*C[:,:,1]*v*dlk[:,:,0,1] + \
           Cst[:,:,0]*C[:,:,0]*v*dlk[:,:,0,0] + \
           Cst[:,:,1]*C[:,:,1]*v*dlk[:,:,1,1] + \
           Cst[:,:,1]*C[:,:,0]*v*dlk[:,:,1,0]
    return term


def CalcPotTerm4(pops, v, dlk):
    return v * ((pops[:, :, 0] * f[:,:,0]) + \
                 pops[:, :, 1] * f[:,:,1])


t3 = CalcPotTerm3(C, v, dlk)



potBO = EhrenPot(pops, E)
pot2 = CalcPotTerm2(C, t).imag
pot3 = CalcPotTerm3(C, v, dlk).imag * 1j
pot4 = CalcPotTerm4(pops, v, f)

potE = potBO # + pot2 #- pot3 #- pot4
kinE = v**2 * 0.5 * m



plt.figure()
#plt.plot(t, potBO, 'r', lw=0.7)
plt.plot(t, potBO + pot2 + pot3 + pot4, 'b', lw=0.7)
plt.plot(t, kinE, 'r', lw=0.7)
plt.plot(t, potE + kinE, 'k', lw=0.7, alpha=0.3)
plt.plot(t, np.mean(potE + kinE, axis=1), 'k--')
plt.show()


raise SystemExit
plt.figure()
plt.plot(t, potBO + kinE, 'r', lw=0.7)
plt.plot(t, kinE, 'b--', lw=0.7)
plt.plot(t, potBO, 'k--', lw=0.7)
plt.xlabel("Time [au]")
plt.ylabel("Energy [Ha]")

plt.figure()
plt.plot(t, np.mean(Fqm + Feh, axis=1), 'r', lw=1)
plt.plot(t, np.mean(Feh, axis=1), 'b--', lw=0.7)
plt.plot(t, np.mean(Fqm, axis=1), 'k--', lw=0.7)
plt.xlabel("Time [au]")
plt.ylabel("Force [au_f]")
plt.show()


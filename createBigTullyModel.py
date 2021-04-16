import hamiltonian as ham_ut
import matplotlib.pyplot as plt
import numpy as np


create_H1 = ham_ut.create_H1
create_H2 = ham_ut.create_H2
create_H3 = ham_ut.create_H3
create_H4 = ham_ut.create_H4


def constantHighCouplings(x):
    return np.array([[0.03, 0.03],
                     [0.03, -0.03]])


def createManyCross(x):
    if x > 100:
        x %= 100

    if x < 20:
        return create_H1(x)

    elif 20 <= x < 40:
        return -create_H1(x-40)

    elif 40 <= x < 80:
        return -create_H1(x-40)

    else:
        return create_H1(x-100)


def createCmplx(x):
    if x >= 100:
        x %= 100
        x -= 20

    if x < 20:
        return create_H1(x)

    elif 20 <= x < 60:
        H = create_H4(x-40, A=0.03, D=5, B=0.01)
        H[0,1]*=-1
        H[1,0]*=-1
        return H

    elif 60 <= x < 100:
        H = create_H2(x-80, E0=0.06)
        H[0,0] += 0.03
        H[1,1] = -H[1, 1] + 0.03
        H[1,1], H[0,0] = H[0,0], H[1,1]
        return -H

    else:
        return create_H1(x)


def plot_H(x, Ham):
    adE = np.array([np.linalg.eigh(i)[0] for i in Ham])

    plt.plot(x, adE[:, 0])
    plt.plot(x, adE[:, 1])

    #plt.plot(x, Ham[:, 0, 0])
    #plt.plot(x, Ham[:, 0, 1])
    #plt.plot(x, Ham[:, 1, 1])
    #plt.plot(x, Ham[:, 1, 0])

    #plt.xlim([79, 81])
    #plt.ylim([0.1+0.1797e-5, 0.1+0.18e-5])

    plt.show()



x = np.arange(-20, 240, 0.01)
H = np.array([constantHighCouplings(i) for i in x])

plot_H(x, H)

from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:41:34 2019

@author: mellis
"""

import numpy as np


def create_H1(x, A=0.03, B=0.4, C=0.005, D=0.3):
    """
    Will create the Hamiltonian in Tully Model 1
    """
    V11 = A*np.tanh(B*x)

    V22 = -V11

    V12 = C * np.exp(-D*(x**2))

    return np.matrix([[V11, V12], [V12, V22]])


def create_H2(x, A=0.1, B=0.28, C=0.015, D=0.06, E0=0.05):
    """
    Will create the Hamiltonian in Tully Model 2
    """
    V11 = 0
    V22 = -A * np.exp(-B*(x**2)) + E0
    V12 = C*np.exp(-D*(x**2))

    return np.matrix([[V11, V12], [V12, V22]])


def create_H3(x, A=6e-4, B=0.1, C=0.9):
    """
    Will create the Hamiltonian in Tully Model 3
    """
    V11 = A
    if x <= 0:
        V12 = B*np.exp(C*x)
    else:
        V12 = B*(2-np.exp(-C*x))
    V22 = -V11
    return np.matrix([[V11, V12], [V12, V22]])


def create_H4(x, A=6e-4, B=0.1, C=0.9, D=4):
    """
    Will create the Hamiltonian in Tully Model 3
    """
    V11 = A
    V22 = -V11
    if x <= -D:
        V12 = B * (-np.exp(C *(x-D)) + np.exp(C *(x+D)))
    elif x >= D:
        V12 = B * (np.exp(-C *(x-D)) - np.exp(-C *(x+D)))
    else:
        V12 = B * (2 - np.exp(C *(x-D)) - np.exp(-C *(x+D)))

    return np.matrix([[V11, V12],
                      [V12, V22]])


def create_Hlin(x, slope=-0.01, Start=-15, Egap=0.05):
   """
   Will create a linearly decreasing Hamiltonian with 0 coupling
   """
   V11 = slope * (x - Start)
   V22 = Egap + (slope * (x - Start))
   return np.matrix([[V11, 0],
                     [0, V22]])


def constantHighCouplings(x):
    return np.matrix([[0.03, 0.03],
                      [0.03, -0.03]])


def createManyCross(x):
    """
    Will create repeated avoided crossings
    """
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


def createBigHam(x):
    if x < 20:
        return create_H1(x)

    elif 20 <= x < 60:
        return create_H4(x-40, A=0.03, D=5)

    elif 60 <= x:
        H = create_H2(x-80, E0=0.06)
        H[0,0] += 0.03
        H[1,1] = -H[1, 1] + 0.03
        H[1,1], H[0,0] = H[0,0], H[1,1]
        return -H


def getEigProps(H, ctmqc_env):
    """
    Wrapper function, this really needs taking out it when I have time.
    """
    return np.linalg.eigh(H)


def calcNACVgradPhi(pos, ctmqc_env):
    """
    Will use a different method to calculate the NACV. This function will
    simply use:
        d = <phil | grad phik>

    This is slightly more unstable because sometimes the eigen solver sometimes
    mixes up the order of the eigenvectors/eigenvalues which causes large
    differences in phi for different pos.
    """
    dx = ctmqc_env['dx']
#    H_xm = ctmqc_env['Hfunc'](pos - dx)
    H_x = ctmqc_env['Hfunc'](pos)
    H_xp = ctmqc_env['Hfunc'](pos + dx)
#    nstate = len(H_x)

    allU = [np.linalg.eigh(H)[1]
            for H in (H_xp, H_x)]
#            for H in (H_xm, H_x, H_xp)]
#    gradU = np.gradient(allU, dx, axis=0)
    gradU = (allU[1] - allU[0]) / dx

    NACV = np.zeros((2, 2))
    for l in range(2):
        for k in range(l):
#            print((allU[1].A[l], gradU.A[k]))
            NACV[l, k] = np.dot(allU[1].A[l], gradU.A[k])
            NACV[k, l] = -NACV[l, k]
#    print(NACV)

    # Check the anti-symettry of the NACV
    badNACV = False
    for l in range(len(NACV)):
        for k in range(l+1, len(NACV)):
            if np.abs(NACV[l, k] + np.conjugate(NACV[k, l])) > 1e-10:
                badNACV = True

    if badNACV:
        print("gradPhi NACV bad switching to gradH")
        NACV = calcNACVgradH(pos, ctmqc_env)
        NACV = 0.5*(NACV - NACV.T)

    return NACV


def calcNACV(irep, ctmqc_env):
    """
    If we are using model 2 low momentum then use the gradPhi NACV
    """
    pos = ctmqc_env['pos'][irep]

    return calcNACVgradPhi(pos, ctmqc_env)

   # if ctmqc_env['tullyModel'] == 2 and ctmqc_env['velInit'] < 0.01:
   # else:
   #     return calcNACVgradH(pos, ctmqc_env)

def calcNACVgradH(pos, ctmqc_env):
    """
    Will calculate the adiabatic NACV for replica irep
    """
    dx = ctmqc_env['dx']
    nState = ctmqc_env['nstate']

    H_xm = ctmqc_env['Hfunc'](pos - dx)
    H_x = ctmqc_env['Hfunc'](pos)
    H_xp = ctmqc_env['Hfunc'](pos + dx)

    allH = [H_xm, H_x, H_xp]
    gradH = np.gradient(allH, dx, axis=0)[1]
    E, U = getEigProps(H_x, ctmqc_env)
    NACV = np.zeros((nState, nState), dtype=complex)
    for l in range(nState):
        for k in range(nState):
            if l != k:
                phil = np.array(U)[l]
                phik = np.array(U)[k]
                NACV[l, k] = np.dot(phil, np.dot(gradH, phik))
                NACV[l, k] /= E[k] - E[l]

    for l in range(len(NACV)):
        for k in range(l+1, len(NACV)):
            if np.abs(NACV[l, k] + np.conjugate(NACV[k, l])) > 1e-10:
                print("NACV:")
                print(NACV)
                print("NACV[%i, %i]: " % (l, k), NACV[l, k])
                print("NACV[%i, %i]*: " % (l, k), np.conjugate(NACV[k, l]))
                raise SystemExit("NACV not antisymetric!")

    NACV = 0.5*(NACV - NACV.T)
    return NACV


def test_Hfunc(Hfunc, minX=-15, maxX=15, stride=0.01):
    import matplotlib.pyplot as plt
    allR = np.arange(minX, maxX, stride)
    allH = [Hfunc({}, x) for x in allR]
    allE = [np.linalg.eigh(H)[0] for H in allH]
    allE = np.array(allE)
    plt.plot(allR, allE[:, 0], 'g', label=r"E$_{1}^{ad}$")
    plt.plot(allR, allE[:, 1], 'b', label=r"E$_{1}^{ad}$")
    plt.show()
    return allH, allE


def test_NACV(Hfunc):
    nrep = 4000
    randomNums = (np.random.random(nrep) * 30) - 15
    ctmqc_env = {'dx': 1e-5, 'nstate': 2, 'pos': list(sorted(randomNums)),
                 'Hfunc': Hfunc}

    allNACV1 = [calcNACV(irep, ctmqc_env) for irep in range(nrep)]
    allNACV2 = [calcNACVgradH(irep, ctmqc_env)
                for irep in range(nrep)]

    allNACV1 = np.array(allNACV1)
    allNACV2 = np.array(allNACV2)
    diff = np.abs(allNACV1 - allNACV2)

    worstCase = np.max(diff)
    bestCase = np.min(diff)
    avgCase = np.mean(diff)
    std = np.std(diff)

    #import matplotlib.pyplot as plt

    print("Worst Case: {0}".format(worstCase))
    print("Best Case: {0}".format(bestCase))
    print("Mean Case: {0} +/- {1}".format(avgCase, std))

    #plt.plot(ctmqc_env['pos'], allNACV1[:, 0, 1], 'k.',
    #         label="calcNACV")
    #plt.plot(ctmqc_env['pos'], allNACV2[:, 0, 1], 'y.',
    #         label="calcNACVgradPhi")
    #plt.legend()
    #plt.show()


#test_Hfunc(create_Hlin)
#test_NACV(create_Hlin)
#test_NACV(create_H1)
#test_NACV(create_H2)
#test_NACV(create_H3)
#test_NACV(create_H4)

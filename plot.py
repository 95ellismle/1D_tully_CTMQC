#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:36:24 2019

@author: mellis
"""
import matplotlib.pyplot as plt
import numpy as np

import elec_prop
import hamiltonian as Ham


def plot_ad_pops(x, allAdPops):
    """
    Will plot the adiabatic population output of the simulation.
    """
    plt.plot(x, allAdPops[:, 0, 0], label=r"$|C_{1}|$")
    plt.plot(x, allAdPops[:, 0, 1], label=r"$|C_{2}|$")
    plt.xlabel("Nucl. Crds")
    plt.ylabel("Adiab Pops")
    plt.legend()
    plt.show()


def plot_di_pops(x, allu, xlabel="Nucl. Crds"):
    """
    Will plot the diabatic populations against x
    """
    nstep, nrep, nstate = allu.shape
    allPops = [elec_prop.calc_ad_pops(allu[istep, 0])
               for istep in range(nstep)]
    allPops = np.array(allPops)
    plt.plot(x, allPops[:, 0], label=r"u$_{1}$")
    plt.plot(x, allPops[:, 1], label=r"u$_{2}$")
    plt.xlabel(xlabel)
    plt.ylabel("Diab Pops")
    plt.legend()
    plt.show()


def plot_H(x, allH, xlabel="Nucl. Crds"):
    """
    Will plot each element of H
    """
    plt.plot(x, allH[:, 0, 0, 0], label=r"H$_{1,1}$")
    plt.plot(x, allH[:, 0, 0, 1], label=r"H$_{1,2}$")
    plt.plot(x, allH[:, 0, 1, 0], label=r"H$_{2,1}$")
    plt.plot(x, allH[:, 0, 1, 1], label=r"H$_{2,2}$")

    plt.xlabel(xlabel)
    plt.ylabel("H [Ha]")
    plt.legend()


def plot_Rabi(t, H):
    """
    Will plot Rabi Oscilations for times t using Hamiltonian H
    """
    Hab = H[0, 1]
    delE = H[1, 1] - H[0, 0]
    Hab4 = 4*Hab**2

    omegaR = np.sqrt(delE**2 + Hab4)

    pops = Hab4 / (delE**2 + Hab4)
    pops *= np.sin(0.5*omegaR*t)**2
    pops = 1 - pops

    plt.plot(t, pops, 'k--', lw=2)


def plot_x_t(x, t):
    """
    Will plot the position vs time
    """
    pass

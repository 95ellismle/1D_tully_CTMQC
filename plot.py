#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:36:24 2019

@author: mellis
"""
import matplotlib.pyplot as plt
import numpy as np

import elec_prop
import nucl_prop
import hamiltonian as Ham


def plot_ad_pops(x, allAdPops, params={}):
    """
    Will plot the adiabatic population output of the simulation.
    """
    plt.plot(x, allAdPops[:, :, 0], **params)
    plt.plot(x, allAdPops[:, :, 1], **params)
    plt.xlabel("Nucl. Crds")
    plt.ylabel("Adiab Pops")
#    plt.legend()
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
    plt.plot(x, allH[:, 0, 0], label=r"H$_{1,1}$")
    plt.plot(x, allH[:, 0, 1], label=r"H$_{1,2}$")
    plt.plot(x, allH[:, 1, 0], label=r"H$_{2,1}$")
    plt.plot(x, allH[:, 1, 1], label=r"H$_{2,2}$")

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

    plt.plot(t, pops, 'k--', lw=2, label="Rabi")
    plt.legend()


def plot_x_t(data):
    """
    Will plot the position vs time
    """
    plt.plot(data.allt, data.allR)
    plt.ylabel("Nucl. Pos. [bohr]")
    plt.ylabel("Time. [au_t]")
    plt.show()


def plot_H_all_x(ctmqc_env):
    """
    Willl plot each element of H vs R
    """
    x = np.arange(-15, 15, 0.01)
    allH = np.array([ctmqc_env['Hfunc'](i) for i in x])

    plt.plot(x, allH[:, 0, 0], label=r"H$_{1,1}$")
    plt.plot(x, allH[:, 0, 1], label=r"H$_{1,2}$")
    plt.plot(x, allH[:, 1, 0], label=r"H$_{2,1}$")
    plt.plot(x, allH[:, 1, 1], label=r"H$_{2,2}$")

    plt.xlabel("Nucl. Crds.")
    plt.ylabel("H")
    plt.legend()


def plot_ener_all_x(ctmqc_env):
    """
    Will plot the energy states for x between -15, 15
    """
    x = np.arange(-15, 15, 0.01)
    allH = [ctmqc_env['Hfunc'](i) for i in x]

    eigProps = [Ham.getEigProps(H, ctmqc_env) for H in allH]
    allE = np.array([i[0] for i in eigProps])

    plt.plot(x, allE[:, 0], label=r"E$_1$")
    plt.plot(x, allE[:, 1], label=r"E$_2$")
    plt.xlabel("Nucl. Coords [Bohr]")
    plt.ylabel("Energy [Ha]")
    plt.legend()
    plt.show()


def plot_eh_frc_all_x(ctmqc_env, label=False):
    """
    Will plot the Ehrenfest force for x between -15, 15 with the coefficient
    given in the ctmqc_env. This coefficient is constant.
    """
    allF = []
    allX = []
    irep = 0
    for pos in np.arange(-15, 15, 0.02):
        ctmqc_env['pos'][0] = pos
        ctmqc_env['H'][irep] = ctmqc_env['Hfunc'](ctmqc_env['pos'][irep])
        gradE = nucl_prop.calc_ad_frc(pos, ctmqc_env)
        ctmqc_env['adFrc'][irep] = gradE

        pop = elec_prop.calc_ad_pops(ctmqc_env['C'][irep],
                                     ctmqc_env)
        ctmqc_env['adPops'][irep] = pop

        F = nucl_prop.calc_ehren_adiab_force(irep, gradE, pop, ctmqc_env)
        allF.append(F)
        allX.append(pos)

    allF = np.array(allF)
    allX = np.array(allX)

    if label is False:
        plt.plot(allX, allF, label=r"$\mathbf{F}_{ehren}$")
    else:
        plt.plot(allX, allF, label=label)

    plt.xlabel("Nucl. Crds")
    plt.ylabel("Force")
    plt.legend()
    plt.show()


def plot_NACV_all_x(ctmqc_env, multiplier=1, params={}):
    """
    Will plot the NACV for x between -15, 15
    """
    allNACV = []
    allX = []
    for x in np.arange(-15, 15, 0.02):
        ctmqc_env['pos'][0] = x
        NACV = Ham.calcNACV(0, ctmqc_env)
        allX.append(x)
        allNACV.append(NACV)

    allNACV = np.array(allNACV) * multiplier
    allX = np.array(allX)

    plt.plot(allX, allNACV[:, 0, 0], label=r"$\mathbf{d}_{11}$", **params)
    plt.plot(allX, allNACV[:, 0, 1], label=r"$\mathbf{d}_{12}$", **params)
    plt.plot(allX, allNACV[:, 1, 0], label=r"$\mathbf{d}_{21}$", **params)
    plt.plot(allX, allNACV[:, 1, 1], label=r"$\mathbf{d}_{22}$", **params)
    plt.xlabel("Nucl. Crds")
    plt.ylabel("NACV")
    plt.legend()
    plt.show()


def plot_adFrc_all_x(ctmqc_env):
    """
    Will plot the adiabatic forces for x between -15, 15
    """
    x = np.arange(-15, 15, 0.01)
    allAdFrc = np.array([nucl_prop.calc_ad_frc(i, ctmqc_env)
                         for i in x])

    plt.plot(x, allAdFrc[:, 0], label=r"$\nabla$E$_1$")
    plt.plot(x, allAdFrc[:, 1], label=r"$\nabla$E$_2$")
    plt.xlabel("Nucl. Coords [Bohr]")
    plt.ylabel("Energy [Ha]")
    plt.legend()
    plt.show()


def plot_adStates_all_x(ctmqc_env):
    """
    Will plot the adiabatic states for x between -15, 15
    """
    x = np.arange(-15, 15, 0.01)
    allH = [ctmqc_env['Hfunc'](i) for i in x]

    eigProps = [Ham.getEigProps(H, ctmqc_env) for H in allH]
    allU = np.array([i[1] for i in eigProps])

    plt.plot(x, allU[:, 0, 0], label=r"U$_{1, 1}$")
    plt.plot(x, allU[:, 0, 1], label=r"U$_{1, 2}$")
    plt.plot(x, allU[:, 1, 0], label=r"U$_{2, 1}$")
    plt.plot(x, allU[:, 1, 1], label=r"U$_{2, 2}$")
    plt.xlabel("Nucl. Coords [Bohr]")
    plt.ylabel("")
    plt.legend()
    plt.show()


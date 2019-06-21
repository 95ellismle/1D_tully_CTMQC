#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:40:43 2019

Remember to check propagator by turning off forces and checking dx/dt

@author: mellis
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
import random as rd

import hamiltonian as Ham
import nucl_prop
import elec_prop
import plot

nRep = 1
v_mean = 5e-3 * 1.6
v_std = 0#5e-4

pos = [-8 for i in range(nRep)]
vel = [rd.gauss(v_mean, v_std) for i in range(nRep)]
coeff = [[complex(1, 0), complex(0, 0)] for i in range(nRep)]

# All units must be atomic units
ctmqc_env = {
        'pos': pos,  # Intial Nucl. pos | nrep |in bohr
        'vel': vel,  # Initial Nucl. veloc | nrep |au_v
        'u': coeff,  # Intial WF |nrep, 2| -
        'mass': [2000],  # nuclear mass |nrep| au_m
        'tullyModel': 2,  # Which model | | -
        'max_time': 2500,  # How many steps | | -
        'dx': 1e-5,  # The increment for the NACV and grad E calc | | bohr
        'dt': 1,  # The timestep | |au_t
        'elec_steps': 10,  # Num elec. timesteps per nucl. one | | -
            }

elecProp = elec_prop.elecProp(ctmqc_env)


class main(object):
    """
    Will carry out the full propagation from intialisation to end.
    """
    allX = []
    allT = []

    def __init__(self, ctmqc_env):
        self.ctmqc_env = ctmqc_env
        self.__init_tully_model()
        self.__init_nsteps()
        self.__init_pos_vel_wf()
        self.__init_arrays()

        self.__init_step()
        self.__main_loop()

    def __init_nsteps(self):
        """
        Will calculate the number of steps from the maximum time needed and the
        timestep.
        """
        max_time = self.ctmqc_env['max_time']
        dt = self.ctmqc_env['dt']
        nsteps = max_time / dt
        self.ctmqc_env['nsteps'] = int(nsteps)

    def __init_pos_vel_wf(self):
        """
        Will get the number of replicas and atoms from the size of the R, v
        and u arrays. Will also check they are there and convert them to numpy
        arrays
        """
        changes = {'replicas': False}
        # Check coeff array
        if 'u' in self.ctmqc_env:
            self.adiab_diab = "diab"
            self.ctmqc_env['u'] = np.array(self.ctmqc_env['u'])
            nrep, nstate = np.shape(self.ctmqc_env['u'])
        elif 'C' in ctmqc_env:
            self.adiab_diab = "adiab"
            self.ctmqc_env['C'] = np.array(self.ctmqc_env['C'])
            nrep, nstate = np.shape(self.ctmqc_env['C'])
        else:
            msg = "Can't find initial wavefunction\n\t"
            msg += "(specify this as 'u' or 'C')"
            raise SystemExit(msg)
        if nstate != 2:
            raise SystemExit("The models currently only work with 2 states")

        # Check pos array
        if 'pos' in self.ctmqc_env:
            self.ctmqc_env['pos'] = np.array(self.ctmqc_env['pos'],
                                             dtype=np.float64)
            nrep1 = len(self.ctmqc_env['pos'])
            nrep = np.min([nrep1, nrep])
            if nrep != nrep1:
                changes['replicas'] = 'coeff & pos'
        else:
            msg = "Can't find initial positionss\n\t"
            msg += "(specify this as 'pos')"
            raise SystemExit(msg)

        # Check pos array
        if 'vel' in self.ctmqc_env:
            self.ctmqc_env['vel'] = np.array(self.ctmqc_env['vel'],
                                             dtype=np.float64)
            nrep1 = len(self.ctmqc_env['vel'])
            nrep = np.min([nrep1, nrep])
            if nrep != nrep1:
                changes['replicas'] = 'velocity & pos'
        else:
            msg = "Can't find initial positionss\n\t"
            msg += "(specify this as 'pos')"
            raise SystemExit(msg)

        for T in changes:
            if changes[T] is not False:
                print("\n\nWARNING: Not all arrays have same num of replicas")
                print("Changing size of arrays so num rep is consistent\n\n")
                self.ctmqc_env['pos'] = self.ctmqc_env['pos'][:nrep]
                self.ctmqc_env['vel'] = self.ctmqc_env['vel'][:nrep]
                if self.adiab_diab == 'adiab':
                    self.ctmqc_env['C'] = self.ctmqc_env['C'][:nrep]
                else:
                    self.ctmqc_env['u'] = self.ctmqc_env['u'][:nrep]

        self.ctmqc_env['nrep'] = nrep
        self.ctmqc_env['nstate'] = nstate

        print("Number Replicas = %i" % nrep)

    def __init_arrays(self):
        """
        Will fill the ctmqc_env dictionary with the correct sized arrays such
        as the force array
        """
        nrep = self.ctmqc_env['nrep']
        nstate, nstep = self.ctmqc_env['nstate'], self.ctmqc_env['nsteps']
        if 'mass' in self.ctmqc_env:
            self.ctmqc_env['mass'] = np.array(self.ctmqc_env['mass'])
        else:
            raise SystemExit("Mass not specified in startup")

        # For saving the data
        self.allX = np.zeros((nstep, nrep))
        self.allv = np.zeros((nstep, nrep))
        self.allT = np.zeros((nstep))
        self.allE = np.zeros((nstep, nrep, nstate))
        self.allC = np.zeros((nstep, nrep, nstate), dtype=complex)
        self.allu = np.zeros((nstep, nrep, nstate), dtype=complex)
        self.allAdPop = np.zeros((nstep, nrep, nstate))
        self.allH = np.zeros((nstep, nrep, nstate, nstate))

        # For propagating dynamics
        self.ctmqc_env['frc'] = np.zeros((nrep))
        self.ctmqc_env['F_eh'] = np.zeros((nrep))
        self.ctmqc_env['F_ctmqc'] = np.zeros((nrep))
        self.ctmqc_env['acc'] = np.zeros((nrep))
        self.ctmqc_env['H'] = np.zeros((nrep, nstate, nstate))
        self.ctmqc_env['U'] = np.zeros((nrep, nstate, nstate))
        self.ctmqc_env['E'] = np.zeros((nrep, nstate))
        self.ctmqc_env['adFrc'] = np.zeros((nrep, nstate))
        self.ctmqc_env['adPops'] = np.zeros((nrep, nstate))

    def __init_tully_model(self):
        """
        Will put the correct tully model in the ctmqc_env dict
        """
        if self.ctmqc_env['tullyModel'] == 1:
            self.ctmqc_env['Hfunc'] = Ham.create_H1
        elif self.ctmqc_env['tullyModel'] == 2:
            self.ctmqc_env['Hfunc'] = Ham.create_H2
        elif self.ctmqc_env['tullyModel'] == 3:
            self.ctmqc_env['Hfunc'] = Ham.create_H3
        else:
            print("Tully Model = %i" % self.ctmqc_env['tullyModel'])
            msg = "Incorrect tully model chosen. Only 1, 2 and 3 available"
            raise SystemExit(msg)

    def __update_vars_step(self):
        """
        Will update the time-dependant variables in the ctmqc environment

        N.B. Only pos needs saving as the rest are re-calculated on the fly.
        """
        self.ctmqc_env['pos_tm'] = copy.deepcopy(self.ctmqc_env['pos'])

    def __init_step(self):
        """
        Will carry out the initialisation step (just 1 step without
        wf propagation for RK4)
        """
        nrep = self.ctmqc_env['nrep']

        # Calculate the Hamiltonian
        for irep in range(nrep):
            pos = self.ctmqc_env['pos'][irep]
            self.ctmqc_env['H'][irep] = self.ctmqc_env['Hfunc'](pos)

        # Transform the coefficieints
        if 'u' in self.ctmqc_env:
            self.ctmqc_env['C'] = np.zeros((nrep, 2), dtype=complex)
            for irep in range(nrep):
                C = Ham.trans_diab_to_adiab(self.ctmqc_env['H'][irep],
                                            self.ctmqc_env['u'][irep],
                                            self.ctmqc_env)
                self.ctmqc_env['C'][irep] = C
        else:
            self.ctmqc_env['u'] = np.zeros((nrep, 2), dtype=complex)
            for irep in range(nrep):
                u = Ham.trans_adiab_to_diab(self.ctmqc_env['H'][irep],
                                            self.ctmqc_env['C'][irep],
                                            self.ctmqc_env)
                self.ctmqc_env['u'][irep] = u

        for irep in range(nrep):
            pos = self.ctmqc_env['pos'][irep]
            adFrc = nucl_prop.calc_ad_frc(pos, self.ctmqc_env)
            self.ctmqc_env['adFrc'][irep] = adFrc

            pop = elec_prop.calc_ad_pops(self.ctmqc_env['C'][irep],
                                         self.ctmqc_env)
            self.ctmqc_env['adPops'][irep] = pop

            F = nucl_prop.calc_ehren_adiab_force(irep, adFrc, pop, ctmqc_env)
            self.ctmqc_env['F_eh'] = F
            self.ctmqc_env['acc'] = F/ctmqc_env['mass'].astype(float)

            self.ctmqc_env['frc'] = F

        self.ctmqc_env['t'] = 0
        self.ctmqc_env['iter'] = 0
        self.__update_vars_step()

    def __main_loop(self):
        """
        Will loop over all steps and propagate the dynamics
        """
        nstep = self.ctmqc_env['nsteps']
        for istep in range(nstep):
            self.__save_data()
            self.__ctmqc_step()
            self.ctmqc_env['t'] += self.ctmqc_env['dt']
            self.ctmqc_env['iter'] += 1
            print("\rStep %i/%i" % (self.ctmqc_env['iter'], nstep), end="\r")
        self.__finalise()

    def __ctmqc_step(self):
        """
        Will carry out a single step in the CTMQC.
        """
        dt = self.ctmqc_env['dt']
        nrep = self.ctmqc_env['nrep']

        self.ctmqc_env['vel'] += 0.5 * self.ctmqc_env['acc'] * dt  # half dt
        self.ctmqc_env['pos'] += self.ctmqc_env['vel']*dt  # full dt

        for irep in range(nrep):
            pos = self.ctmqc_env['pos'][irep]
            self.ctmqc_env['H'][irep] = self.ctmqc_env['Hfunc'](pos)

            if self.adiab_diab == 'adiab':
                elecProp.do_adiab_prop(irep)
            else:
                elecProp.do_diab_prop(irep)
                C = Ham.trans_diab_to_adiab(self.ctmqc_env['H'][irep],
                                            self.ctmqc_env['u'][irep],
                                            self.ctmqc_env)
                self.ctmqc_env['C'][irep] = C

            pos = self.ctmqc_env['pos'][irep]
            gradE = nucl_prop.calc_ad_frc(pos, self.ctmqc_env)
            self.ctmqc_env['adFrc'][irep] = gradE

            pop = elec_prop.calc_ad_pops(self.ctmqc_env['C'][irep],
                                         self.ctmqc_env)
            self.ctmqc_env['adPops'][irep] = pop

            F = nucl_prop.calc_ehren_adiab_force(irep, gradE, pop, ctmqc_env)
            self.ctmqc_env['F_eh'] = F
            self.ctmqc_env['acc'] = F/ctmqc_env['mass'].astype(float)

            self.ctmqc_env['frc'] = F

        self.ctmqc_env['vel'] += 0.5 * self.ctmqc_env['acc'] * dt  # full dt

        self.__update_vars_step()  # Save old positions

    def __save_data(self):
        """
        Will save data to RAM (arrays within this class)
        """
        istep = self.ctmqc_env['iter']
        self.allX[istep] = self.ctmqc_env['pos']
        self.allv[istep] = self.ctmqc_env['vel']
        self.allE[istep] = self.ctmqc_env['E']
        self.allC[istep] = self.ctmqc_env['C']
        self.allu[istep] = self.ctmqc_env['u']
        self.allAdPop[istep] = self.ctmqc_env['adPops']
        self.allH[istep] = self.ctmqc_env['H']

        self.allT[istep] = self.ctmqc_env['t']

    def __finalise(self):
        """
        Will tidy things up, change types of storage arrays to numpy arrays.
        """
        self.allX = np.array(self.allX)
        self.allT = np.array(self.allT)

    def plot_avg_vel(self):
        """
        Will plot x vs t and fit a linear line through it to get avg velocity
        """
        x = self.allX[:, 0, 0]
        t = self.allT
        fit = np.polyfit(t, x, 1)
        print(fit[0])
        plt.plot(t, x)
        plt.plot(t, np.polyval(fit, t))




data = main(ctmqc_env)

f, axes = plt.subplots(2)

R = data.allX[:, 0]
axes[0].plot(data.allT, data.allAdPop[:, :, 0], 'r')
axes[0].plot(data.allT, data.allAdPop[:, :, 1], 'b')


deco = data.allAdPop[:, :, 0] * data.allAdPop[:, :, 1]
axes[1].plot(data.allT, deco)

f2, axes2 = plt.subplots(2)
norm = np.sum(data.allAdPop, axis=2)
axes2[0].plot(data.allT, norm)

kinE = 0.5 * data.ctmqc_env['mass'][0] * (data.allv**2)
potE = np.sum(data.allAdPop * data.allE, axis=2)
axes2[1].plot(data.allT, kinE + potE, 'k')
axes2[1].plot(data.allT, kinE, 'r')
axes2[1].plot(data.allT, potE, 'g')


#plot.plot_di_pops(data.allT, data.allu, "Time")
#plot.plot_Rabi(data.allT, data.allH[0, 0], ctmqc_env)
#plot.plot_ad_pops(R, data.allAdPop)
#plot.plot_H(data.allT, data.allH, "Time")

#plot.plot_H_all_x(ctmqc_env)
#plot.plot_eh_frc_all_x(ctmqc_env)
#plot.plot_adFrc_all_x(ctmqc_env)
#plot.plot_ener_all_x(ctmqc_env)
#plot.plot_NACV_all_x(ctmqc_env)

from __future__ import print_function
"""
Created on Thu May  9 14:40:43 2019

Remember to check propagator by turning off forces and checking dx/dt

@author: mellis
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rd
import datetime as dt
import pickle
import time
import collections

import hamiltonian as Ham
import nucl_prop
import elec_prop as e_prop
import plot
import QM_utils as qUt

redo = True
whichPlot = '|C|^2 deco'
velMultiplier = 2.5
maxTime = 2000
model = 1
p_mean = -8
s_mean = 2


nRep = 30
natom = 1
mass = 2000
v_mean = 5e-3 * velMultiplier
v_std = 0  # 2.5e-4 * 0.7
p_std = 10 / (v_mean * mass)
s_std = 0

pos = [[rd.gauss(p_mean, p_std) for v in range(natom)] for I in range(nRep)]
vel = [[abs(rd.gauss(v_mean, v_std)) for v in range(natom)]
       for I in range(nRep)]
coeff = [[[complex(1, 0), complex(0, 0)] for v in range(natom)]
         for i in range(nRep)]

corrV = 1
if np.mean(vel) != 0:
    corrV = v_mean / np.mean(vel)
vel = np.array(vel) * corrV

corrP = 1
if np.mean(pos) != 0:
    corrP = p_mean / np.mean(pos)
pos = np.array(pos) * corrP

sigma = [[rd.gauss(s_mean, s_std) for v in range(natom)] for I in range(nRep)]


def setup(pos, vel, coeff, sigma):
    # All units must be atomic units
    ctmqc_env = {
            'pos': pos,  # Intial Nucl. pos | nrep |in bohr
            'vel': vel,  # Initial Nucl. veloc | nrep |au_v
            'C': coeff,  # Intial WF |nrep, 2| -
            'mass': [mass],  # nuclear mass |nrep| au_m
            'tullyModel': model,  # Which model | | -
            'max_time': maxTime,  # Maximum time to simulate to | | au_t
            'dx': 1e-6,  # The increment for the NACV and grad E calc | | bohr
            'dt': 2,  # The timestep | |au_t
            'elec_steps': 5,  # Num elec. timesteps per nucl. one | | -
            'do_QM_F': True,  # Do the QM force
            'do_QM_C': True,  # Do the QM force
            'do_sigma_calc': False,  # Dynamically adapt the value of sigma
            'sigma': sigma,  # The value of sigma (width of gaussian)
            'const': 12,  # The constant in the sigma calc
                }
    return ctmqc_env


def print_timings(timings_dict, ntabs=0, max_len=50, depth=0):
    """
    Will print timing data in a pretty way.

    Inputs:
        * timings_dict  =>  A dictionary containing all the timings data
        * ntabs         =>  OPTIONAL (not recommended to change). Number of
                            tabs in the printout.
    Ouputs:
        None
    """
    bullets = ['*', '>', '#', '-', '+', '=']

    def print_line(line):
        line = line+" "*(max_len-len(line))
        print(line)

    tab = "    "
    for Tkey in timings_dict:
        if isinstance(timings_dict[Tkey], (dict, collections.OrderedDict)):
            line = "%s%s %s:" % (tab*ntabs, bullets[depth], Tkey)
            print_line(line)
            print_timings(timings_dict[Tkey], ntabs+1, depth=depth+1)
        elif isinstance(timings_dict[Tkey], (list,)):
            line = "%s%s %s:" % (tab*ntabs, bullets[depth], Tkey)
            str_num = "%.3f s" % np.mean(timings_dict[Tkey])
            line = line + " " * (max_len-26 -
                                 (len(line) + len(str_num))
                                 + ntabs*5) + str_num
            print_line(line)
        else:
            line = "%s%s %s:" % (tab*ntabs, bullets[depth], Tkey)
            str_num = "%.3f s" % timings_dict[Tkey]
            line = line + " " * (max_len-26 -
                                 (len(line) + len(str_num))
                                 + ntabs*5) + str_num
            print_line(line)
    return


class CTMQC(object):
    """
    Will carry out the full propagation from intialisation to end.
    """
    allR = []
    allt = []

    def __init__(self, ctmqc_env):
        # Set everything up
        self.ctmqc_env = ctmqc_env
        self.__init_tully_model()  # Set the correct Hamiltonian function
        self.__init_nsteps()  # Find how many steps to take
        self.__init_pos_vel_wf()  # set pos vel wf as arrays, get nrep & natom
        self.__init_arrays()  # Create the arrays used
        self.__init_sigma()  # Will initialise the nuclear width

        # Carry out the propagation
        self.__init_step()  # Get things prepared for RK4 (propagate positions)
        self.__main_loop()  # Loop over all steps and propagate
        self.__finalise()  # Finish up and tidy

    def __init_nsteps(self):
        """
        Will calculate the number of steps from the maximum time needed and the
        timestep.
        """
        max_time = self.ctmqc_env['max_time']
        dt = self.ctmqc_env['dt']
        nsteps = int(max_time // dt)
        self.ctmqc_env['nsteps'] = nsteps

    def __init_sigma(self):
        """
        Init the nuclear width
        """
        nrep, natom = self.ctmqc_env['nrep'], self.ctmqc_env['natom']
        if isinstance(self.ctmqc_env['sigma'], float):
            self.ctmqc_env['sigma'] = [self.ctmqc_env['sigma']] * natom
            self.ctmqc_env['sigma'] = [self.ctmqc_env['sigma']
                                       for i in range(nrep)]

        self.ctmqc_env['sigma'] = np.array(self.ctmqc_env['sigma'])
        self.ctmqc_env['sigma'] = self.ctmqc_env['sigma'][:nrep, :natom]
        self.ctmqc_env['sigma_tm'] = np.array(self.ctmqc_env['sigma'])
        self.ctmqc_env['const'] = float(self.ctmqc_env['const'])

    def __check_pos_vel_QM(self):
        """
        Checks whether there is any variation in the positions and velocities
        if the quantum momentum term is being used because if all positions are
        the same then the QM is 0.
        """
        if self.ctmqc_env['do_QM_F'] or self.ctmqc_env['do_QM_C']:
            v_std = np.std(self.ctmqc_env['vel'])
            p_std = np.std(self.ctmqc_env['pos'])
            if p_std == 0 and v_std == 0:
                print("\n\n\nWARNING\n\n")
                print("The initial positions and velocities are all the same,")
                print(" meaning that the quantum momentum will be 0. If this ")
                print("is OK then ignore this warning.\n\n")
                print("WARNING\n\n\n")

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
            nrep, natom, nstate = np.shape(self.ctmqc_env['u'])
        elif 'C' in self.ctmqc_env:
            self.adiab_diab = "adiab"
            self.ctmqc_env['C'] = np.array(self.ctmqc_env['C'])
            nrep, natom, nstate = np.shape(self.ctmqc_env['C'])
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
            nrep1, natom1 = np.shape(self.ctmqc_env['pos'])

            if natom != natom1:
                changes['atoms'] = "velocity & pos"
            natom = np.min([natom, natom1])
            if nrep != nrep1:
                changes['replicas'] = 'coeff & pos'
            nrep = np.min([nrep1, nrep])
        else:
            msg = "Can't find initial positions\n\t"
            msg += "(specify this as 'pos')"
            raise SystemExit(msg)

        # Check pos array
        if 'vel' in self.ctmqc_env:
            self.ctmqc_env['vel'] = np.array(self.ctmqc_env['vel'],
                                             dtype=np.float64)
            nrep1, natom1 = np.shape(self.ctmqc_env['vel'])

            if nrep != nrep1:
                changes['replicas'] = 'velocity & pos'
            nrep = np.min([nrep1, nrep])
            if natom != natom1:
                changes['atoms'] = "velocity & pos"
            natom = np.min([natom, natom1])
        else:
            msg = "Can't find initial velocities\n\t"
            msg += "(specify this as 'vel')"
            raise SystemExit(msg)

        for T in changes:
            if changes[T] is not False:
                print("\n\nWARNING: Not all arrays have same num of %s" % T)
                print("Changing size of arrays so num %s is consistent\n" % T)
                print("\n")
                self.ctmqc_env['pos'] = self.ctmqc_env['pos'][:nrep]
                self.ctmqc_env['vel'] = self.ctmqc_env['vel'][:nrep]
                if self.adiab_diab == 'adiab':
                    self.ctmqc_env['C'] = self.ctmqc_env['C'][:nrep]
                else:
                    self.ctmqc_env['u'] = self.ctmqc_env['u'][:nrep]

        if natom > 1:
            msg = "\n\nSTOP!\n"
            msg += "The code is currently not ready for more than 1 atom\n\n"
            raise SystemExit(msg)

        self.ctmqc_env['nrep'] = nrep
        self.ctmqc_env['nstate'] = nstate
        self.ctmqc_env['natom'] = natom

        print("Number Replicas = %i" % nrep)
        print("Number Atoms = %i" % natom)
        self.__check_pos_vel_QM()  # Just check that the QM will be non-zero

    def __init_arrays(self):
        """
        Will fill the ctmqc_env dictionary with the correct sized arrays such
        as the force array
        """
        nrep, natom = self.ctmqc_env['nrep'], self.ctmqc_env['natom']
        nstate, nstep = self.ctmqc_env['nstate'], self.ctmqc_env['nsteps']
        if 'mass' in self.ctmqc_env:
            self.ctmqc_env['mass'] = np.array(self.ctmqc_env['mass'])
        else:
            raise SystemExit("Mass not specified in startup")

        # For saving the data
        self.allR = np.zeros((nstep, nrep, natom))
        self.allF = np.zeros((nstep, nrep, natom))
        self.allFeh = np.zeros((nstep, nrep, natom))
        self.allFqm = np.zeros((nstep, nrep, natom))
        self.allt = np.zeros((nstep))
        self.allv = np.zeros((nstep, nrep, natom))
        self.allE = np.zeros((nstep, nrep, natom, nstate))
        self.allC = np.zeros((nstep, nrep, natom, nstate), dtype=complex)
        self.allu = np.zeros((nstep, nrep, natom, nstate), dtype=complex)
        self.allAdPop = np.zeros((nstep, nrep, natom, nstate))
        self.allH = np.zeros((nstep, nrep, natom, nstate, nstate))
        self.allAdMom = np.zeros((nstep, nrep, natom, nstate))
        self.allAdFrc = np.zeros((nstep, nrep, natom, nstate))
#        self.allQM = np.zeros((nstep, nrep, natom))
        self.allQlk = np.zeros((nstep, nrep, natom, nstate, nstate))
        self.allSigma = np.zeros((nstep, nrep, natom))

        # For propagating dynamics
        self.ctmqc_env['frc'] = np.zeros((nrep, natom))
        self.ctmqc_env['F_eh'] = np.zeros((nrep, natom))
        self.ctmqc_env['F_qm'] = np.zeros((nrep, natom))
        self.ctmqc_env['acc'] = np.zeros((nrep, natom))
        self.ctmqc_env['H'] = np.zeros((nrep, natom, nstate, nstate))
        self.ctmqc_env['NACV'] = np.zeros((nrep, natom, nstate, nstate),
                                          dtype=complex)
        self.ctmqc_env['NACV_tm'] = np.zeros((nrep, natom, nstate, nstate),
                                             dtype=complex)
        self.ctmqc_env['U'] = np.zeros((nrep, natom, nstate, nstate))
        self.ctmqc_env['E'] = np.zeros((nrep, natom, nstate))
        self.ctmqc_env['adFrc'] = np.zeros((nrep, natom, nstate))
        self.ctmqc_env['adPops'] = np.zeros((nrep, natom, nstate))
        self.ctmqc_env['adMom'] = np.zeros((nrep, natom, nstate))
        self.ctmqc_env['adMom_tm'] = np.zeros((nrep, natom, nstate))
#        self.ctmqc_env['QM'] = np.zeros((nrep, natom))
#        self.ctmqc_env['QM_tm'] = np.zeros((nrep, natom))
        self.ctmqc_env['Qlk'] = np.zeros((nrep, natom, nstate, nstate))
        self.ctmqc_env['Qlk_tm'] = np.zeros((nrep, natom, nstate, nstate))

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
        self.ctmqc_env['vel_tm'] = copy.deepcopy(self.ctmqc_env['vel'])
        self.ctmqc_env['Qlk_tm'] = copy.deepcopy(self.ctmqc_env['Qlk'])
        self.ctmqc_env['NACV_tm'] = copy.deepcopy(self.ctmqc_env['NACV'])
        self.ctmqc_env['adMom_tm'] = copy.deepcopy(self.ctmqc_env['adMom'])
        self.ctmqc_env['sigma_tm'] = np.array(self.ctmqc_env['sigma'])

    def __init_step(self):
        """
        Will carry out the initialisation step (just 1 step without
        wf propagation for RK4)
        """
        nrep, natom = self.ctmqc_env['nrep'], self.ctmqc_env['natom']
        nstate = self.ctmqc_env['nstate']

        # Calculate the Hamiltonian (why not over natom too?)
        for irep in range(nrep):
            for v in range(natom):
                pos = self.ctmqc_env['pos'][irep, v]
                self.ctmqc_env['H'][irep, v] = self.ctmqc_env['Hfunc'](pos)

        # Transform the coefficieints
        if 'u' in self.ctmqc_env:
            self.ctmqc_env['C'] = np.zeros((nrep, natom, nstate),
                                           dtype=complex)
            C = e_prop.trans_diab_to_adiab(
                                          self.ctmqc_env['H'],
                                          self.ctmqc_env['u'],
                                          self.ctmqc_env)
            self.ctmqc_env['C'] = C
        else:
            self.ctmqc_env['u'] = np.zeros((nrep, natom, nstate),
                                           dtype=complex)

            u = e_prop.trans_adiab_to_diab(self.ctmqc_env['H'],
                                           self.ctmqc_env['u'],
                                           self.ctmqc_env)
            self.ctmqc_env['u'] = u

        # Calculate the QM, adMom, adPop, adFrc.
        self.__calc_quantities()
        for irep in range(nrep):
            for v in range(natom):
                # Calculate the forces
                self.__calc_F()

        self.ctmqc_env['t'] = 0
        self.ctmqc_env['iter'] = 0
        self.__update_vars_step()

    def __calc_quantities(self):
        """
        Will calculate the various paramters to feed into the force and
        electronic propagators. These are then saved in the ctmqc_env dict.
        """
        # Do for each rep
        for irep in range(self.ctmqc_env['nrep']):
            for v in range(self.ctmqc_env['natom']):
                # Get Hamiltonian
                pos = self.ctmqc_env['pos'][irep, v]
                self.ctmqc_env['H'][irep, v] = self.ctmqc_env['Hfunc'](pos)

                # Get adiabatic forces
                gradE = qUt.calc_ad_frc(pos, self.ctmqc_env)
                self.ctmqc_env['adFrc'][irep, v] = gradE

                # Get adiabatic populations
                pop = e_prop.calc_ad_pops(self.ctmqc_env['C'][irep, v],
                                          self.ctmqc_env)
                self.ctmqc_env['adPops'][irep, v] = pop

                # Get adiabatic NACV
                self.ctmqc_env['NACV'][irep, v] = Ham.calcNACV(irep, v,
                                                               self.ctmqc_env)

                # Get the QM quantities
                if self.ctmqc_env['do_QM_F'] or self.ctmqc_env['do_QM_C']:
                    if any(Ck > 0.995 for Ck in pop):
                        adMom = 0.8 * self.ctmqc_env['adMom'][irep, v]
                    else:

                        adMom = qUt.calc_ad_mom(self.ctmqc_env, irep, v,
                                                gradE)
                    self.ctmqc_env['adMom'][irep, v] = adMom

        # Do for all reps
        if self.ctmqc_env['do_QM_F'] or self.ctmqc_env['do_QM_C']:
            if self.ctmqc_env['do_sigma_calc']:
                qUt.calc_sigma(self.ctmqc_env)
#            for irep in range(self.ctmqc_env['nrep']):
#                for v in range(self.ctmqc_env['natom']):
#                   QM = qUt.calc_QM_analytic(ctmqc_env, irep, v)
#                   self.ctmqc_env['QM'][irep, v] = QM
            self.ctmqc_env['Qlk'] = qUt.calc_Qlk(self.ctmqc_env)

    def __main_loop(self):
        """
        Will loop over all steps and propagate the dynamics
        """
        nstep = self.ctmqc_env['nsteps']
        self.allTimes = {'step': [], 'force': [], 'wf_prop': {'prop': {
                                                             'makeX': [],
                                                             'RK4': [],
                                                             'lin. interp': [],
                                                              },
                                                              'transform': []},
                         'prep': []}

        for istep in range(nstep):
            try:
                t1 = time.time()
                self.__save_data()
                self.__ctmqc_step()
                self.ctmqc_env['t'] += self.ctmqc_env['dt']
                self.ctmqc_env['iter'] += 1

                t2 = time.time()

                # Print some useful info
                self.allTimes['step'].append(t2 - t1)
                avgTime = np.mean(self.allTimes['step'])
                msg = "\rStep %i/%i  Time Taken = %.2gs" % (istep, nstep,
                                                            avgTime)
                timeLeft = int((nstep - istep) * avgTime)
                timeLeft = str(dt.timedelta(seconds=timeLeft))
                msg += "  Time Left = %s" % (timeLeft)
                percentComplete = (float(istep) / float(nstep)) * 100
                msg += "  %i%% Complete" % (percentComplete)
    #            print(" "*200, end="\r")
                print(msg,
                      end="\r")
            except KeyboardInterrupt:
                print("\nOk Exiting Safely")
                return

    def __calc_F(self):
        """
        Will calculate the force on the nuclei
        """
        for irep in range(self.ctmqc_env['nrep']):
            for v in range(self.ctmqc_env['natom']):
                # Get Ehrenfest Forces
                Feh = nucl_prop.calc_ehren_adiab_force(
                                             irep, v,
                                             self.ctmqc_env['adFrc'][irep, v],
                                             self.ctmqc_env['adPops'][irep, v],
                                             self.ctmqc_env)

                Fqm = 0.0
                if self.ctmqc_env['do_QM_F']:
                    Qlk = self.ctmqc_env['Qlk'][irep, v, 0, 1]
                    Fqm = nucl_prop.calc_QM_force(
                                             self.ctmqc_env['adPops'][irep, v],
                                             Qlk,
                                             self.ctmqc_env['adMom'][irep, v],
                                             self.ctmqc_env)

                Ftot = float(Feh) + float(Fqm)
                self.ctmqc_env['F_eh'][irep, v] = Feh
                self.ctmqc_env['F_qm'][irep, v] = Fqm
                self.ctmqc_env['frc'][irep, v] = Ftot
                self.ctmqc_env['acc'][irep, v] = Ftot/self.ctmqc_env['mass'][0]

    def __prop_wf(self):
        """
        Will propagate the wavefunction in the correct basis and transform the
        coefficients.
        """
        # Propagate WF
#        t1 = time.time()
        if self.ctmqc_env['do_QM_C']:
            if self.adiab_diab == 'adiab':
                e_prop.do_adiab_prop_QM(self.ctmqc_env, self.allTimes)
            else:
                e_prop.do_diab_prop_QM(self.ctmqc_env)
        else:
            if self.adiab_diab == 'adiab':
                e_prop.do_adiab_prop_ehren(self.ctmqc_env)
            else:
                e_prop.do_diab_prop_ehren(self.ctmqc_env)
        t2 = time.time()

        # Transform WF
        if self.adiab_diab == 'adiab':
            # if self.ctmqc_env['iter'] % 30 == 0:
            #   self.ctmqc_env['C'] = e_prop.renormalise_all_coeffs(
            #   self.ctmqc_env['C'])
            u = e_prop.trans_adiab_to_diab(self.ctmqc_env['H'],
                                           self.ctmqc_env['C'],
                                           self.ctmqc_env)
            self.ctmqc_env['u'] = u
        else:
            if self.ctmqc_env['iter'] % 30 == 0:
                self.ctmqc_env['u'] = e_prop.renormalise_all_coeffs(
                                                   self.ctmqc_env['u'])
            C = e_prop.trans_adiab_to_diab(self.ctmqc_env['H'],
                                           self.ctmqc_env['u'],
                                           self.ctmqc_env)
            self.ctmqc_env['C'] = C
        t3 = time.time()

        self.allTimes['wf_prop']['transform'].append(t3 - t2)

    def __ctmqc_step(self):
        """
        Will carry out a single step in the CTMQC.
        """
        dt = self.ctmqc_env['dt']

        self.ctmqc_env['vel'] += 0.5 * self.ctmqc_env['acc'] * dt  # half dt
        self.ctmqc_env['pos'] += self.ctmqc_env['vel']*dt  # full dt

        t1 = time.time()
        self.__calc_quantities()
        t2 = time.time()

        self.__prop_wf()
        t3 = time.time()
        self.__calc_F()
        t4 = time.time()

        self.allTimes['prep'].append(t2 - t1)
#        self.allTimes['wf_prop'].append(t3 - t2)
        self.allTimes['force'].append(t4 - t3)
        self.__update_vars_step()  # Save old positions

        self.__update_vars_step()  # Save old positions

    def __save_data(self):
        """
        Will save data to RAM (arrays within this class)
        """
        istep = self.ctmqc_env['iter']
        self.allR[istep] = self.ctmqc_env['pos']
        self.allF[istep] = self.ctmqc_env['frc']
        self.allFeh[istep] = self.ctmqc_env['F_eh']
        self.allFqm[istep] = self.ctmqc_env['F_qm']
        self.allE[istep] = self.ctmqc_env['E']
        self.allC[istep] = self.ctmqc_env['C']
        self.allu[istep] = self.ctmqc_env['u']
        self.allAdPop[istep] = self.ctmqc_env['adPops']
        self.allH[istep] = self.ctmqc_env['H']
        self.allAdMom[istep] = self.ctmqc_env['adMom']
        self.allAdFrc[istep] = self.ctmqc_env['adFrc']
        self.allv[istep] = self.ctmqc_env['vel']
#        self.allQM[istep] = self.ctmqc_env['QM']
        self.allQlk[istep] = self.ctmqc_env['Qlk']
        self.allSigma[istep] = self.ctmqc_env['sigma']

        self.allt[istep] = self.ctmqc_env['t']

    def __finalise(self):
        """
        Will tidy things up, change types of storage arrays to numpy arrays.
        """
        self.allR = np.array(self.allR)
        self.allt = np.array(self.allt)

        self.allR = self.allR[:self.ctmqc_env['iter']]
        self.allt = self.allt[:self.ctmqc_env['iter']]
        self.allF = self.allF[:self.ctmqc_env['iter']]
        self.allFeh = self.allFeh[:self.ctmqc_env['iter']]
        self.allFqm = self.allFqm[:self.ctmqc_env['iter']]
        self.allE = self.allE[:self.ctmqc_env['iter']]
        self.allC = self.allC[:self.ctmqc_env['iter']]
        self.allu = self.allu[:self.ctmqc_env['iter']]
        self.allAdPop = self.allAdPop[:self.ctmqc_env['iter']]
        self.allH = self.allH[:self.ctmqc_env['iter']]
        self.allAdMom = self.allAdMom[:self.ctmqc_env['iter']]
        self.allAdFrc = self.allAdFrc[:self.ctmqc_env['iter']]
        self.allv = self.allv[:self.ctmqc_env['iter']]
        self.allQlk = self.allQlk[:self.ctmqc_env['iter']]
        self.allSigma = self.allSigma[:self.ctmqc_env['iter']]

        # Print some useful info
        sumTime = np.sum(self.allTimes['step'])
        nstep = self.ctmqc_env['iter']
        msg = "\r                                                             "
        msg += "                                                              "
        msg += "\n\n***\n"
        timeTaken = np.ceil(sumTime)
        timeTaken = str(dt.timedelta(seconds=timeTaken))
        msg += "Steps = %i   Total Time Taken__prop_wf = %ss" % (nstep, timeTaken)
        msg += "  Avg. Time Per Step = %.2gs" % np.mean(self.allTimes['step'])
        msg += "  All Done!\n***\n"

        msg += "\n\nAverage Times:"
        print(msg)
        print_timings(self.allTimes, 1)

    def plot_avg_vel(self):
        """
        Will plot x vs t and fit a linear line through it to get avg velocity
        """
        x = self.allR[:, 0, 0]
        t = self.allt
        fit = np.polyfit(t, x, 1)
        print(fit[0])
        plt.plot(t, x)
        plt.plot(t, np.polyval(fit, t))


sigma = [[rd.gauss(s_mean, s_std) for v in range(natom)] for I in range(nRep)]
if redo:
    ctmqc_env = setup(pos, vel, coeff, sigma)
    data = CTMQC(ctmqc_env)

    with open('data', 'wb') as f:
        pickle.dump(data, f)

else:
    with open('data', 'rb') as f:
        data = pickle.load(f)

"""
    ###   Now Plot the Data   ###
"""

if data.ctmqc_env['iter'] < 30:
    whichPlot = ""

# Actually do the plotting
if isinstance(whichPlot, str):
    whichPlot = whichPlot.lower()
    if 'qm_fl_fk' in whichPlot:
        fig, axes = plt.subplots(2)
        # Plot the QM and adiabatic momentum

        ax = axes[0]
        fl_fk = (data.allAdMom[:, :, 0, 0] - data.allAdMom[:, :, 0, 1])
        QM = data.allQlk[:, :, 0, 0, 1] * data.ctmqc_env['mass'][0]
        fl_fk *= QM

        def init_QM_fl_fk():
            minR, maxR = np.min(data.allR), np.max(data.allR)
            minflQ, maxflQ = np.min(fl_fk), np.max(fl_fk)
            minQ, maxQ = np.min(QM), np.max(QM)

            ln1, = ax.plot(np.linspace(minR, maxR, QM.shape[1]),
                           np.linspace(minflQ, maxflQ, QM.shape[1]), 'y.')
            ln2, = ax.plot(np.linspace(minR, maxR, QM.shape[1]),
                           np.linspace(minQ, maxQ, QM.shape[1]),
                           '.', color=(1, 0, 1))
            return ln1, ln2

        def ani_init_2():
            return ln1, ln2

        def animate_QM_fl_fk(i):
            step = i % len(data.allt)
            ln1.set_ydata(fl_fk[step, :])  # update the data.
            ln1.set_xdata(data.allR[step, :, 0])
            ln2.set_ydata(QM[step, :])  # update the data.
            ln2.set_xdata(data.allR[step, :, 0])
            return ln1, ln2

        ln1, ln2 = init_QM_fl_fk()
        axes[1].plot(data.allR[:, :, 0], data.allE[:, 0, 0])

        ani = animation.FuncAnimation(
                fig, animate_QM_fl_fk, init_func=ani_init_2,
                interval=10, blit=True)

        plt.show()

    # Nuclear Density
    if whichPlot == 'nucl_dens':
        nstep, nrep, natom = np.shape(data.allR)
        allND = [qUt.calc_nucl_dens_PP(R, sig)
                 for R, sig in zip(data.allR, data.allSigma)]
        allND = np.array(allND)
        minR, maxR = np.min(data.allR), np.max(data.allR)
        minND, maxND = np.min(allND[:, 0]), np.max(allND[:, 0])
        f, axes = plt.subplots(1)

        ln, = axes.plot(np.linspace(minR, maxR, nrep),
                        np.linspace(minND, maxND, nrep),
                        '.', color=(1., 0., 1.))

        def ani_init_1():
            for I in range(nrep):
                axes.plot(data.allR[:, I, 0], data.allE[:, I, 0],
                          lw=0.3, alpha=0.2)
            return ln,

        def animate_ND(i):
            step = i % nstep
            ln.set_ydata(allND[step][0])
            ln.set_xdata(allND[step][1])
            return ln,

        ani = animation.FuncAnimation(
                f, animate_ND, init_func=ani_init_1, interval=10, blit=True)

        plt.show()

    R = data.allR[:, 0]
    if '|c|^2' in whichPlot:
        plt.figure()
        # Plot ad coeffs
        for I in range(data.ctmqc_env['nrep']):
            params = {'lw': 0.5, 'alpha': 0.1, 'color': 'r'}
            plot.plot_ad_pops(data.allt, data.allAdPop[:, I, 0, 0], params)
            params = {'lw': 0.5, 'alpha': 0.1, 'color': 'b'}
            plot.plot_ad_pops(data.allt, data.allAdPop[:, I, 0, 1], params)

        avgData = np.mean(data.allAdPop, axis=1)
        params = {'lw': 2, 'alpha': 1, 'ls': '--', 'color': 'r'}
        plot.plot_ad_pops(data.allt, avgData[:, 0, 0], params)
        params = {'lw': 2, 'alpha': 1, 'ls': '--', 'color': 'b'}
        plot.plot_ad_pops(data.allt, avgData[:, 0, 1], params)
        plt.xlabel("Time [au_t]")
        plt.annotate(r"K$_0$ = %.1f au" % (v_mean * data.ctmqc_env['mass'][0]),
                     (10, 0.5), fontsize=24)

#        plt.title("Sigma = %.2f" % s_mean)
#        plt.savefig("/home/oem/Documents/PhD/Graphs/1D_Tully/Model1/CTMQC_25K/ChangingSigma/%.2f_pops.png" % s_mean)

    if 'deco' in whichPlot:
        # Plot Decoherence
        plt.figure()
        allDeco = data.allAdPop[:, :, :, 0] * data.allAdPop[:, :, :, 1]
        avgDeco = np.mean(allDeco, axis=1)
        plt.plot(data.allt, avgDeco)

        minD, maxD = np.min(avgDeco), np.max(avgDeco)
        rD = maxD - minD
        plt.annotate(r"K$_0$ = %.1f au" % (v_mean * data.ctmqc_env['mass'][0]),
                     (10, minD+(rD/2.)), fontsize=24)
        plt.ylabel("Decoherence")
        plt.xlabel("Time [au_t]")

#        plt.title("Sigma = %.2f" % s_mean)
#        plt.savefig("/home/oem/Documents/PhD/Graphs/1D_Tully/Model1/CTMQC_25K/ChangingSigma/%.2f_deco.png" % s_mean)

    if '|u|^2' in whichPlot:
        plt.figure()
        plot.plot_di_pops(data.allt, data.allu, "Time")
    if 'norm' in whichPlot:
        norm = np.linalg.norm(data.allC, axis=3)
        plt.figure()
        plt.plot(data.allt, norm[:, :, 0], lw=0.7, alpha=0.5)
        plt.plot(data.allt, np.mean(norm[:, :, 0], axis=1), lw=1.3, ls='--')
        plt.xlabel("Time [au_t]")
        plt.ylabel("Norm (adPops)")
    if 'rabi' in whichPlot:
        plot.plot_Rabi(data.allt, data.allH[0, 0])
    if 'ham' in whichPlot:
        plot.plot_H(data.allt, data.allH, "Time")

    if 'ham_ax' in whichPlot:
        plot.plot_H_all_x(data.ctmqc_env)

    if 'eh_frc_ax' in whichPlot:
        for i in np.arange(0, 1, 0.1):
            pops = [1-i, i]
            data.ctmqc_env['C'] = np.array([[complex(np.sqrt(1-i), 0),
                                             complex(np.sqrt(i), 0)]])
            plot.plot_eh_frc_all_x(data.ctmqc_env, label=r"$|C|^2 = $%.1g" % i)
    if 'ad_frc_ax' in whichPlot:
        plot.plot_adFrc_all_x(data.ctmqc_env)
    if 'ad_ener_ax' in whichPlot:
        plot.plot_ener_all_x(data.ctmqc_env)
    if 'nacv_ax' in whichPlot:
        plot.plot_NACV_all_x(data.ctmqc_env)

    plt.show()
#    plt.close("all")
from __future__ import print_function
"""
Created on Thu May  9 14:40:43 2019

Remember to check propagator by turning off forces and checking dx/dt

@author: mellis
"""
import numpy as np
import copy
import matplotlib.pyplot as plt
import random as rd
import datetime as dt
import time
import os
import collections

import hamiltonian as Ham
import nucl_prop
import elec_prop as e_prop
import QM_utils as qUt
import plot

numRepeats = 10

all_velMultiplier = [4, 2, 3, 1, 3, 1.6, 2.5, 1] * numRepeats
all_maxTime = [2000, 2500, 1300, 5500, 1500, 2500, 2000, 3500] * numRepeats
all_model = [4, 4, 3, 3, 2, 2, 1, 1] * numRepeats
all_p_mean = [-15, -15, -15, -15, -8, -8, -8, -8] * numRepeats
all_doCTMQC_C = ([True] * 8) * numRepeats
all_doCTMQC_F = ([True] * 8 )  * numRepeats
rootFolder = '/scratch/mellis/TullyModelData/EffectiveR/Repeat'

#all_velMultiplier = [1] * numRepeats
#all_maxTime = [3500] * numRepeats
#all_model = [1] * numRepeats
#all_p_mean = [-8] * numRepeats
#all_doCTMQC_C = [True] * numRepeats
#all_doCTMQC_F = [True]  * numRepeats
#rootFolder = '/scratch/mellis/TullyModelData/Dev'

s_mean = 0.3
nRep = 200
mass = 2000


nSim = min([len(all_velMultiplier), len(all_maxTime),
            len(all_model), len(all_p_mean), len(all_doCTMQC_C),
            len(all_doCTMQC_F)])
coeff = [[complex(1, 0), complex(0, 0)]
         for i in range(nRep)]


def setup(pos, vel, coeff, sigma, maxTime, model, doCTMQC_C, doCTMQC_F):
    # All units must be atomic units
    ctmqc_env = {
            'pos': pos,  # Intial Nucl. pos | nrep |in bohr
            'vel': vel,  # Initial Nucl. veloc | nrep |au_v
            'C': coeff,  # Intial WF |nrep, 2| -
            'mass': [mass],  # nuclear mass |nrep| au_m
            'tullyModel': model,  # Which model | | -
            'max_time': maxTime,  # Maximum time to simulate to | | au_t
            'dx': 1e-5,  # The increment for the NACV and grad E calc | | bohr
            'dt': 1,  # The timestep | |au_t
            'elec_steps': 5,  # Num elec. timesteps per nucl. one | | -
            'do_QM_F': doCTMQC_F,  # Do the QM force
            'do_QM_C': doCTMQC_C,  # Do the QM force
            'do_sigma_calc': False,  # Dynamically adapt the value of sigma
            'sigma': sigma,  # The value of sigma (width of gaussian)
            'const': 15,  # The constant in the sigma calc
            'nSmoothStep': 5,  # The number of steps to take to smooth the QM intercept
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

    def __init__(self, ctmqc_env, root_folder = False):
        
        # Set everything up
        self.root_folder = root_folder
        self.ctmqc_env = ctmqc_env

        self.__create_folderpath()
        self.__init_tully_model()  # Set the correct Hamiltonian function
        self.__init_nsteps()  # Find how many steps to take
        self.__init_pos_vel_wf()  # set pos vel wf as arrays, get nrep
        self.__init_arrays()  # Create the arrays used
        self.__init_sigma()  # Will initialise the nuclear width

        # Carry out the propagation
        self.__init_step()  # Get things prepared for RK4 (propagate positions)
        self.__main_loop()  # Loop over all steps and propagate
        self.__finalise()  # Finish up and tidy
    
    def __create_folderpath(self):
        """
        Will determine where to store the data.
        """
        if bool(self.root_folder) is False:
            self.saveFolder = False
            return
        self.root_folder = os.path.abspath(self.root_folder)
        
        eHStr = "Ehren"
        if self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
            eHStr = "CTMQC"
        elif not self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
            eHStr = "CTMQCF_EhC"
        elif self.ctmqc_env['do_QM_C'] and not self.ctmqc_env['do_QM_F']:
            eHStr = "CTMQCC_EhF"

        modelStr = "Model_%i" % self.ctmqc_env['tullyModel']
        mom = np.round(self.ctmqc_env['vel'][0] * self.ctmqc_env['mass'][0])
        momStr = "Kinit_%i" % int(mom)
        if self.ctmqc_env['do_sigma_calc']:
            sigStr = "varSig"
        else:
            sigStr = "%.2gSig" % self.ctmqc_env['sigma'][0]
        self.saveFolder = "%s/%s/%s/%s" % (self.root_folder, eHStr,
                                              modelStr, momStr)
        
        # This should be recursive but I can't be bothered making it work in a
        #   class.
        count = 0
        rootFold = self.root_folder[:]
        while (os.path.isdir(self.saveFolder)):
            self.root_folder = "%s_%i" % (rootFold, count)
            if self.root_folder is False:
                self.root_folder = os.getcwd()
            self.root_folder = os.path.abspath(self.root_folder)
            
            eHStr = "Ehren"
            if self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
                eHStr = "CTMQC"
            elif not self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
                eHStr = "CTMQCF_EhC"
            elif self.ctmqc_env['do_QM_C'] and not self.ctmqc_env['do_QM_F']:
                eHStr = "CTMQCC_EhF"
    
            modelStr = "Model_%i" % self.ctmqc_env['tullyModel']
            mom = np.round(self.ctmqc_env['vel'][0] * self.ctmqc_env['mass'][0])
            momStr = "Kinit_%i" % int(mom)
            if self.ctmqc_env['do_sigma_calc']:
                sigStr = "varSig"
            else:
                sigStr = "%.2gSig" % self.ctmqc_env['sigma'][0]
            self.saveFolder = "%s/%s/%s/%s" % (self.root_folder, eHStr,
                                                  modelStr, momStr)
            count += 1
        
        try:
            os.makedirs(self.saveFolder)
        except OSError:
            if bool(self.root_folder) is False:
                self.saveFolder = False
                return
            self.root_folder = os.path.abspath(self.root_folder)
            
            eHStr = "Ehren"
            if self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
                eHStr = "CTMQC"
            elif not self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
                eHStr = "CTMQCF_EhC"
            elif self.ctmqc_env['do_QM_C'] and not self.ctmqc_env['do_QM_F']:
                eHStr = "CTMQCC_EhF"
    
            modelStr = "Model_%i" % self.ctmqc_env['tullyModel']
            mom = np.round(self.ctmqc_env['vel'][0] * self.ctmqc_env['mass'][0])
            momStr = "Kinit_%i" % int(mom)
            if self.ctmqc_env['do_sigma_calc']:
                sigStr = "varSig"
            else:
                sigStr = "%.2gSig" % self.ctmqc_env['sigma'][0]
            self.saveFolder = "%s/%s/%s/%s" % (self.root_folder, eHStr,
                                                  modelStr, momStr)
            
            # This should be recursive but I can't be bothered making it work in a
            #   class.
            count = 0
            rootFold = self.root_folder[:]
            while (os.path.isdir(self.saveFolder)):
                self.root_folder = "%s_%i" % (rootFold, count)
                if self.root_folder is False:
                    self.root_folder = os.getcwd()
                self.root_folder = os.path.abspath(self.root_folder)
                
                eHStr = "Ehren"
                if self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
                    eHStr = "CTMQC"
                elif not self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
                    eHStr = "CTMQCF_EhC"
                elif self.ctmqc_env['do_QM_C'] and not self.ctmqc_env['do_QM_F']:
                    eHStr = "CTMQCC_EhF"
        
                modelStr = "Model_%i" % self.ctmqc_env['tullyModel']
                mom = np.round(self.ctmqc_env['vel'][0] * self.ctmqc_env['mass'][0])
                momStr = "Kinit_%i" % int(mom)
                if self.ctmqc_env['do_sigma_calc']:
                    sigStr = "varSig"
                else:
                    sigStr = "%.2gSig" % self.ctmqc_env['sigma'][0]
                self.saveFolder = "%s/%s/%s/%s/%s" % (self.root_folder, eHStr,
                                                      modelStr, momStr, sigStr)
                count += 1
#        print("\r%s" % self.saveFolder, end="\r")
    
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
        nrep = self.ctmqc_env['nrep']
        if isinstance(self.ctmqc_env['sigma'], float):
            self.ctmqc_env['sigma'] = [self.ctmqc_env['sigma']]
            self.ctmqc_env['sigma'] = [self.ctmqc_env['sigma']
                                       for i in range(nrep)]

        self.ctmqc_env['sigma'] = np.array(self.ctmqc_env['sigma'])
        self.ctmqc_env['sigma'] = self.ctmqc_env['sigma'][:nrep]
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
        changes = False
        # Check coeff array
        if 'u' in self.ctmqc_env:
            self.adiab_diab = "diab"
            self.ctmqc_env['u'] = np.array(self.ctmqc_env['u'])
            nrep, nstate = np.shape(self.ctmqc_env['u'])
        elif 'C' in self.ctmqc_env:
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
            nrep1, = np.shape(self.ctmqc_env['pos'])
            nrep = np.min([nrep1, nrep])
            if nrep != nrep1:
                changes = True
        else:
            msg = "Can't find initial positions\n\t"
            msg += "(specify this as 'pos')"
            raise SystemExit(msg)

        # Check pos array
        if 'vel' in self.ctmqc_env:
            self.ctmqc_env['vel'] = np.array(self.ctmqc_env['vel'],
                                             dtype=np.float64)
            nrep1, = np.shape(self.ctmqc_env['vel'])

            nrep = np.min([nrep1, nrep])
            if nrep != nrep1:
                changes = True
        else:
            msg = "Can't find initial velocities\n\t"
            msg += "(specify this as 'vel')"
            raise SystemExit(msg)
        
        if changes:
            print("\n\nWARNING: Not all arrays have same num of replicas")
            print("Changing size of arrays so num of replicas is consistent\n")
            print("New Num Rep = %i" % nrep)
            print("\n")
            self.ctmqc_env['pos'] = self.ctmqc_env['pos'][:nrep]
            self.ctmqc_env['vel'] = self.ctmqc_env['vel'][:nrep]
            if 'u' in self.ctmqc_env:
                self.ctmqc_env['u'] = self.ctmqc_env['u'][:nrep, :]
            else:
                self.ctmqc_env['C'] = self.ctmqc_env['C'][:nrep, :]

        self.ctmqc_env['nrep'] = nrep
        self.ctmqc_env['nstate'] = nstate

#        print("\n\nNumber Replicas = %i\n\n" % nrep)
        self.__check_pos_vel_QM()  # Just check that the QM will be non-zero

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
        self.allR = np.zeros((nstep, nrep))
        self.allF = np.zeros((nstep, nrep))
        self.allNACV = np.zeros((nstep, nrep, nstate, nstate), dtype=complex)
        self.allFeh = np.zeros((nstep, nrep))
        self.allFqm = np.zeros((nstep, nrep))
        self.allt = np.zeros((nstep))
        self.allv = np.zeros((nstep, nrep))
        self.allE = np.zeros((nstep, nrep, nstate))
        self.allC = np.zeros((nstep, nrep, nstate), dtype=complex)
        self.allu = np.zeros((nstep, nrep, nstate), dtype=complex)
        self.allAdPop = np.zeros((nstep, nrep, nstate))
        self.allH = np.zeros((nstep, nrep, nstate, nstate))
        self.allAdMom = np.zeros((nstep, nrep, nstate))
        self.allAdFrc = np.zeros((nstep, nrep, nstate))
        self.allQlk = np.zeros((nstep, nrep, nstate, nstate))
        self.allAlphal = np.zeros(nstep)
        self.allRlk = np.zeros((nstep, nstate, nstate))
        self.allRI0 = np.zeros((nstep, nrep))
        self.allEffR = np.zeros((nstep))
        self.allSigma = np.zeros((nstep, nrep))
        self.allSigmal = np.zeros((nstep, nstate))
        self.allRl = np.zeros((nstep, nstate))

        # For propagating dynamics
        self.ctmqc_env['frc'] = np.zeros((nrep))
        self.ctmqc_env['F_eh'] = np.zeros((nrep))
        self.ctmqc_env['F_qm'] = np.zeros((nrep))
        self.ctmqc_env['acc'] = np.zeros((nrep))
        self.ctmqc_env['H'] = np.zeros((nrep, nstate, nstate))
        self.ctmqc_env['NACV'] = np.zeros((nrep, nstate, nstate),
                                          dtype=complex)
        self.ctmqc_env['NACV_tm'] = np.zeros((nrep, nstate, nstate),
                                             dtype=complex)
        self.ctmqc_env['U'] = np.zeros((nrep, nstate, nstate))
        self.ctmqc_env['E'] = np.zeros((nrep, nstate))
        self.ctmqc_env['adFrc'] = np.zeros((nrep, nstate))
        self.ctmqc_env['adPops'] = np.zeros((nrep, nstate))
        self.ctmqc_env['adMom'] = np.zeros((nrep, nstate))
        self.ctmqc_env['adMom_tm'] = np.zeros((nrep, nstate))
        self.ctmqc_env['alpha'] = np.zeros((nrep))
        self.ctmqc_env['alphal'] = 0.0
        self.ctmqc_env['sigmal'] = np.zeros(nstate)
        self.ctmqc_env['Rl'] = np.zeros(nstate)
        self.ctmqc_env['Qlk'] = np.zeros((nrep, nstate, nstate))
        self.ctmqc_env['Qlk_tm'] = np.zeros((nrep, nstate, nstate))
        self.ctmqc_env['effR'] = 0.0 
        self.ctmqc_env['Rlk'] = np.zeros((nstate, nstate))
        self.ctmqc_env['RI0'] = np.zeros((nrep))
        self.ctmqc_env['Rlk_tm'] = np.zeros((nstate, nstate))

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
        elif self.ctmqc_env['tullyModel'] == 4:
            self.ctmqc_env['Hfunc'] = Ham.create_H4
        else:
            print("Tully Model = %i" % self.ctmqc_env['tullyModel'])
            msg = "Incorrect tully model chosen. Only 1, 2, 3 and 4 available"
            raise SystemExit(msg)

    def __init_step(self):
        """
        Will carry out the initialisation step (just 1 step without
        wf propagation for RK4)
        """
        nrep = self.ctmqc_env['nrep']
        nstate = self.ctmqc_env['nstate']

        # Calculate the Hamiltonian
        for irep in range(nrep):
            pos = self.ctmqc_env['pos'][irep]
            self.ctmqc_env['H'][irep] = self.ctmqc_env['Hfunc'](pos)
            E, U = np.linalg.eigh(self.ctmqc_env['H'][irep])
            self.ctmqc_env['E'][irep], self.ctmqc_env['U'][irep] = E, U

        # Transform the coefficieints
        if 'u' in self.ctmqc_env:
            self.ctmqc_env['C'] = np.zeros((nrep, nstate),
                                           dtype=complex)
            e_prop.trans_diab_to_adiab(self.ctmqc_env)
        else:
            self.ctmqc_env['u'] = np.zeros((nrep, nstate),
                                           dtype=complex)
            e_prop.trans_adiab_to_diab(self.ctmqc_env)

        # Calculate the QM, adMom, adPop, adFrc.
        self.__calc_quantities()
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
        # Get adiabatic populations
        self.ctmqc_env['adPops'] = (np.conjugate(self.ctmqc_env['C']) * self.ctmqc_env['C']).real
        
        # Do for each rep
        for irep in range(self.ctmqc_env['nrep']):
            # Get Hamiltonian
            pos = self.ctmqc_env['pos'][irep]
            self.ctmqc_env['H'][irep] = self.ctmqc_env['Hfunc'](pos)

            # Get Eigen properties
            E, U = np.linalg.eigh(self.ctmqc_env['H'][irep])
            self.ctmqc_env['E'][irep], self.ctmqc_env['U'][irep] = E, U

            # Get adiabatic forces
            adFrc = qUt.calc_ad_frc(pos, self.ctmqc_env)
            self.ctmqc_env['adFrc'][irep] = adFrc

            # Get adiabatic NACV
            self.ctmqc_env['NACV'][irep] = Ham.calcNACV(irep,
                                                           self.ctmqc_env)

#            if self.ctmqc_env['pos'][irep] > 0:
#                self.ctmqc_env['NACV'][irep] = -0.1 * self.ctmqc_env['NACV'][irep]

            # Get the QM quantities
            if self.ctmqc_env['do_QM_F'] or self.ctmqc_env['do_QM_C']:
                if any(Ck > 0.995 for Ck in self.ctmqc_env['adPops'][irep]):
                    adMom = 0.8 * self.ctmqc_env['adMom'][irep]
                else:
                    adMom = qUt.calc_ad_mom(self.ctmqc_env, irep, adFrc)
                self.ctmqc_env['adMom'][irep] = adMom

        # Do for all reps
        if self.ctmqc_env['do_QM_F'] or self.ctmqc_env['do_QM_C']:
            if self.ctmqc_env['do_sigma_calc']:
                qUt.calc_sigma(self.ctmqc_env)
#            self.ctmqc_env['Qlk'] = qUt.calc_Qlk(self.ctmqc_env)
            self.ctmqc_env['Qlk'] = qUt.calc_Qlk_2state(self.ctmqc_env)

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
#            try:
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
#            except (KeyboardInterrupt, SystemExit):
#                print("\n\nOk Exiting Safely")
#                return

    def __calc_F(self):
        """
        Will calculate the force on the nuclei
        """
        for irep in range(self.ctmqc_env['nrep']):
            # Get Ehrenfest Forces
            Feh = nucl_prop.calc_ehren_adiab_force(
                                         irep,
                                         self.ctmqc_env['adFrc'][irep],
                                         self.ctmqc_env['adPops'][irep],
                                         self.ctmqc_env)

            Fqm = 0.0
            if self.ctmqc_env['do_QM_F']:
                Qlk = self.ctmqc_env['Qlk'][irep, 0, 1]
                Fqm = nucl_prop.calc_QM_force(
                                         C=self.ctmqc_env['adPops'][irep],
                                         QM=Qlk,
                                         f=self.ctmqc_env['adMom'][irep],
                                         ctmqc_env=self.ctmqc_env)

            Ftot = float(Feh) + float(Fqm)
            self.ctmqc_env['F_eh'][irep] = Feh
            self.ctmqc_env['F_qm'][irep] = Fqm
            self.ctmqc_env['frc'][irep] = Ftot
            self.ctmqc_env['acc'][irep] = Ftot/self.ctmqc_env['mass'][0]

    def __prop_wf(self):
        """
        Will propagate the wavefunction in the correct basis and transform the
        coefficients.
        """
        # Propagate WF
#        t1 = time.time()
        if self.ctmqc_env['do_QM_C']:
            if self.adiab_diab == 'adiab':
                e_prop.do_adiab_prop_QM(self.ctmqc_env)
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
            e_prop.trans_adiab_to_diab(self.ctmqc_env)
        else:
            e_prop.trans_diab_to_adiab(self.ctmqc_env)
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
        self.ctmqc_env['vel'] += 0.5 * self.ctmqc_env['acc'] * dt  # full dt
        t4 = time.time()
        
        self.allTimes['prep'].append(t2 - t1)
#        self.allTimes['wf_prop'].append(t3 - t2)
        self.allTimes['force'].append(t4 - t3)
        self.__update_vars_step()  # Save old positions

    def __update_vars_step(self):
        """
        Will update the time-dependant variables in the ctmqc environment
        """
#        self.ctmqc_env['pos_tm'] = copy.deepcopy(self.ctmqc_env['pos'])
        self.ctmqc_env['H_tm'] = copy.deepcopy(self.ctmqc_env['H'])
        self.ctmqc_env['U_tm'] = copy.deepcopy(self.ctmqc_env['U'])
        if self.adiab_diab == "adiab":
            self.ctmqc_env['vel_tm'] = copy.deepcopy(self.ctmqc_env['vel'])
            self.ctmqc_env['NACV_tm'] = copy.deepcopy(self.ctmqc_env['NACV'])
            self.ctmqc_env['E_tm'] = copy.deepcopy(self.ctmqc_env['E'])
        if self.ctmqc_env['do_QM_C']:
            self.ctmqc_env['Qlk_tm'] = copy.deepcopy(self.ctmqc_env['Qlk'])
            self.ctmqc_env['Rlk_tm'] = copy.deepcopy(self.ctmqc_env['Rlk'])
            self.ctmqc_env['adMom_tm'] = copy.deepcopy(self.ctmqc_env['adMom'])
            self.ctmqc_env['sigma_tm'] = np.array(self.ctmqc_env['sigma'])

    def __save_data(self):
        """
        Will save data to RAM (arrays within this class)
        """
        istep = self.ctmqc_env['iter']
        self.allR[istep] = self.ctmqc_env['pos']
        self.allNACV[istep] = self.ctmqc_env['NACV']
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
        self.allQlk[istep] = self.ctmqc_env['Qlk']
        self.allRlk[istep] = self.ctmqc_env['Rlk']
        self.allEffR[istep] = self.ctmqc_env['effR']
        self.allRI0[istep] = self.ctmqc_env['RI0']
        self.allSigma[istep] = self.ctmqc_env['sigma']
        self.allSigmal[istep] = self.ctmqc_env['sigmal']
        self.allRl[istep] = self.ctmqc_env['Rl']
        self.allAlphal[istep] = self.ctmqc_env['alphal']

        self.allt[istep] = self.ctmqc_env['t']

    def __chop_arrays(self):
        """
        Will splice the arrays to the appropriate size (to num steps done)
        """
        self.allR = self.allR[:self.ctmqc_env['iter']]
        self.allt = self.allt[:self.ctmqc_env['iter']]
        self.allNACV = self.allNACV[:self.ctmqc_env['iter']]
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
        self.allRlk = self.allRlk[:self.ctmqc_env['iter']]
        self.allRI0 = self.allRI0[:self.ctmqc_env['iter']]
        self.allEffR = self.allEffR[:self.ctmqc_env['iter']]
        self.allSigma = self.allSigma[:self.ctmqc_env['iter']]
        self.allSigmal = self.allSigmal[:self.ctmqc_env['iter']]
        self.allRl = self.allRl[:self.ctmqc_env['iter']]
        self.allAlphal = self.allAlphal[:self.ctmqc_env['iter']]

    def __store_data(self):
        """
        Will save all the arrays as numpy binary files.
        """
        if not os.path.isdir(self.saveFolder):
            os.makedirs(self.saveFolder)

        names = ["pos", "time", "Ftot", "Feh", "Fqm", "E", "C", "u", "|C|^2",
                "H", "f", "Fad", "vel", "Qlk", "Rlk", "RI0", "sigma", "sigmal",
                "NACV", "Rl"]
        arrs = [self.allR, self.allt, self.allF, self.allFeh, self.allFqm,
                self.allE, self.allC, self.allu, self.allAdPop, self.allH,
                self.allAdMom, self.allAdFrc, self.allv, self.allQlk,
                self.allRlk, self.allRI0, self.allSigma, self.allSigmal,
                self.allNACV, self.allRl]
        for name, arr in zip(names, arrs):
            savepath = "%s/%s" % (self.saveFolder, name)
            np.save(savepath, arr)

    def __checkVV(self):
        """
        Will check the velocity verlet algorithm propagated the dynamics
        correctly by comparing the velocities with the time-derivative
        positions.
        """
        dx_dt = np.gradient(self.allR, self.ctmqc_env['dt'], axis=0)
        diff = np.abs(self.allv - dx_dt)
        worstCase = np.max(diff)
        avgCase = np.mean(diff)
        bestCase = np.min(diff)
        std = np.std(diff)
        if worstCase > 1e-6 and bestCase > 1e-18 or worstCase > 1e-5:
            msg = "ERROR: Velocity Verlet gives bad velocities / positions"
            msg += "\n\t-Time differentiated positions are different to "
            msg += "velocities!"
            msg += "\n\t* Best case: %.2g" % bestCase
            msg += "\n\t* Worst case: %.2g" % worstCase
            msg += "\n\t* Mean case: %.2g +/- %.2g" % (avgCase, std)
            raise SystemExit(msg)

    def __finalise(self):
        """
        Will tidy things up, change types of storage arrays to numpy arrays.
        """
        self.allR = np.array(self.allR)
        self.allt = np.array(self.allt)
        self.__chop_arrays()
        # Small runs are probably tests
        if self.ctmqc_env['iter'] > 30 and self.saveFolder:
            self.__store_data()
        
        # Run tests on data (only after Ehrenfest, CTMQC normally fails!)
        if all([self.ctmqc_env['do_QM_F'], self.ctmqc_env['do_QM_C']]) is False:
            self.__checkVV()
        
        # Print some useful info
#        sumTime = np.sum(self.allTimes['step'])
#        nstep = self.ctmqc_env['iter']
#        msg = "\r                                                             "
#        msg += "                                                              "
#        msg += "\n\n***\n"
#        timeTaken = np.ceil(sumTime)
#        timeTaken = str(dt.timedelta(seconds=timeTaken))
#        msg += "Steps = %i   Total Time Taken__prop_wf = %ss" % (nstep, timeTaken)
#        msg += "  Avg. Time Per Step = %.2gs" % np.mean(self.allTimes['step'])
#        msg += "  All Done!\n***\n"
#
#        msg += "\n\nAverage Times:"
#        print(msg)
        print("Finished. Saving in %s" % self.saveFolder)
#        print_timings(self.allTimes, 1)

    def plot_avg_vel(self):
        """
        Will plot x vs t and fit a linear line through it to get avg velocity
        """
        x = self.allR[:, 0]
        t = self.allt
        fit = np.polyfit(t, x, 1)
        print(fit[0])
        plt.plot(t, x)
        plt.plot(t, np.polyval(fit, t))


    
def doSim(i):
    velMultiplier = all_velMultiplier[i]
    maxTime = all_maxTime[i]
    model = all_model[i]
    p_mean = all_p_mean[i]
    doCTMQC_C = all_doCTMQC_C[i]
    doCTMQC_F = all_doCTMQC_F[i]
    
    v_mean = 5e-3 * velMultiplier
    v_std = 0  # 2.5e-4 * 0.7
    p_std = 20. / float(v_mean * mass)
    s_std = 0
    
    pos = [rd.gauss(p_mean, p_std) for I in range(nRep)]
    pos = [ -8.04910616,  -8.11373539, -10.02259722,  -8.0410751 ,
        -6.66998231,  -9.0567181 ,  -8.49919676,  -8.6986839 ,
        -7.01809389,  -6.57261684,  -9.33079494,  -7.58902002,
        -7.97566433,  -7.63600355,  -7.84910729,  -8.54683989,
        -9.34682712,  -8.56036804,  -7.86155926,  -6.96933734,
        -8.99622009,  -7.25461229,  -7.72256048,  -9.30023776,
        -8.03907912,  -8.13681033,  -8.04740528,  -6.79830708,
        -8.92520023,  -7.37401661,  -7.94191597,  -8.82468894,
        -6.64447847,  -7.34899066,  -6.43411302,  -8.09172623,
        -9.75957376,  -7.25178972,  -8.02000309,  -7.08721693,
        -8.11062182,  -8.91344714,  -8.2054316 ,  -8.05236252,
        -9.22264851,  -8.48595252,  -7.65054049,  -7.760368  ,
        -8.13274096,  -8.60343613,  -6.88641226,  -7.62126653,
        -7.68510144,  -8.01770285,  -8.37370045,  -8.51342775,
        -8.08402372,  -8.22729659,  -6.8469031 ,  -7.4014849 ,
        -8.85602108,  -8.31835222,  -7.28564433,  -8.5622688 ,
        -9.26021825,  -7.35601602,  -6.11515705,  -7.66672686,
        -6.81958268,  -6.41589623,  -7.52783774,  -8.68118761,
        -7.3890811 ,  -8.02177865,  -8.65204357,  -7.52449401,
        -7.4651457 ,  -7.42041555,  -6.84626713,  -7.3157486 ,
        -8.44141106,  -8.58280185,  -6.98057015,  -8.09453005,
       -10.05864752,  -6.64759544,  -8.13316885,  -8.93045412,
        -8.11407663,  -8.48096764,  -7.35877067,  -7.51303533,
        -7.95265122,  -7.98544436,  -7.03725483,  -7.97898984,
        -9.10472252,  -8.50594345,  -6.59632519,  -6.84294243,
        -8.97191828,  -8.80132626,  -9.10372277,  -7.57211689,
        -8.2382156 ,  -7.10910427,  -7.60848644,  -8.027597  ,
        -8.87585921,  -7.6173295 ,  -7.56801169,  -8.08748334,
        -7.47915758,  -8.57133138,  -8.36510635,  -9.43742457,
        -7.86819007,  -8.02356644,  -8.01034112,  -8.52381399,
        -8.38384938,  -7.71170765,  -7.15312275,  -8.57156641,
        -7.01539276,  -8.21050543,  -8.60719841,  -7.63839256,
        -7.92886356,  -7.45050815,  -8.36289096,  -8.35767176,
        -7.97286798,  -8.41936427,  -8.25334918,  -9.72893678,
        -8.30973889,  -9.00211973,  -7.20605429,  -8.05994986,
        -7.26586108,  -9.12406703,  -6.36636985,  -7.02662133,
        -8.67214473,  -7.60847842,  -8.62075487,  -8.09455161,
        -8.58998741,  -6.65712051,  -7.4493979 ,  -8.36679818,
        -8.45220862,  -8.45032979,  -7.26847089,  -7.3108029 ,
        -7.65072679,  -7.77961175,  -8.75619428,  -7.6573155 ,
        -6.80086091,  -9.28857695,  -7.24309605,  -8.10384208,
        -8.46685755,  -8.11818922,  -9.35635903,  -7.68105989,
        -8.94582102,  -8.02648647,  -7.20364015,  -5.77959916,
        -7.63907439,  -8.61686965,  -8.17341883,  -8.52536182,
        -6.95046271,  -6.74453161,  -7.60737621,  -9.06269249,
        -8.00592844,  -6.3506931 ,  -8.8717187 ,  -8.85129328,
        -9.48373918,  -8.35987096,  -8.09275958,  -8.81699219,
        -7.76780734,  -8.45920348,  -7.05180641,  -9.02860055,
        -8.63769035,  -7.38807801,  -9.10239012,  -7.28412053,
        -8.82399235,  -8.95472582,  -7.5737501 ,  -7.34736727]
    
    vel = [abs(rd.gauss(v_mean, v_std)) for I in range(nRep)]
    
    corrV = 1
    if np.mean(vel) != 0:
        corrV = v_mean / np.mean(vel)
    vel = np.array(vel) * corrV
    
    corrP = 1
    if np.mean(pos) != 0:
        corrP = p_mean / np.mean(pos)
    pos = np.array(pos) * corrP
    

    sigma = [rd.gauss(s_mean, s_std) for I in range(nRep)]

    # Now run the simulation
    ctmqc_env = setup(pos, vel, coeff, sigma, maxTime, model,
                      doCTMQC_C, doCTMQC_F)
    return CTMQC(ctmqc_env, rootFolder)
    

if nSim > 1 and nRep > 30:
    import multiprocessing as mp
    
    nProc = min([nSim, 16])
    pool = mp.Pool(nProc)
    print("Doing %i sims with %i processes" % (nSim, nProc))
    pool.map(doSim, range(nSim))
else:
    #import test
    for iSim in range(nSim):
        runData = doSim(iSim)

        #test.vel_is_diff_x(runData)


if nSim == 1 and runData.ctmqc_env['iter'] > 50:
    plot.plotPops(runData)
    plot.plotDeco(runData)
    plot.plotRlk_Rl(runData)
    #plot.plotNorm(runData)
#    plotSigmal(runData)
#    plot.plotEpotTime(runData, range(0, runData.ctmqc_env['iter']),
#                      saveFolder='/scratch/mellis/Pics')
    plt.show()

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

#inputs = "FullCTMQC"
#inputs = "FullCTMQCEhren"
#inputs = "quickFullCTMQC"
#inputs = "quickFullEhren"
#inputs = "MomentumEhren"
inputs = "custom"
#inputs = "custom"

rootSaveFold = "./"

if inputs == "MomentumEhren":
    all_velMultiplier = np.arange(1, 5, 0.03)
    all_maxTime = [(45.) / (v * 5e-3) for v in all_velMultiplier]
    all_model = [4] * len(all_velMultiplier)
    all_p_mean = [-15] * len(all_velMultiplier)
    all_doCTMQC_C = [False] * len(all_velMultiplier)
    all_doCTMQC_F = [False] * len(all_velMultiplier)
    rootFolder = '%s/MomentumRuns' % rootSaveFold
    all_nRep = [1] * len(all_velMultiplier)

elif inputs == "FullCTMQC":
    print("Carrying out full CTMQC testing!")
    numRepeats = 10
    all_velMultiplier = [4, 2, 3, 1, 3, 1.6, 2.5, 1] * numRepeats
    all_maxTime = [2000, 2500, 1300, 5500, 1500, 2500, 2000, 3500] * numRepeats
    all_model = [4, 4, 3, 3, 2, 2, 1, 1] * numRepeats
    all_p_mean = [-15, -15, -15, -15, -8, -8, -8, -8] * numRepeats
    all_doCTMQC_C = ([True] * 8) * numRepeats
    all_doCTMQC_F = ([True] * 8 )  * numRepeats
    rootFolder = '/scratch/mellis/TullyModelData/FullCTMQC/Repeat'
    all_nRep = [200]

elif inputs == "FullCTMQCEhren":
    print("Carrying out all testing (Ehren and CTMQC)!")
    numRepeats = 10
    all_velMultiplier = [4, 2, 3, 1, 3, 1.6, 2.5, 1] * numRepeats * 2
    all_maxTime = [2000, 2500, 1300, 5500, 1500, 2500, 2000, 3500] * numRepeats * 2
    all_model = [4, 4, 3, 3, 2, 2, 1, 1] * numRepeats * 2
    all_p_mean = [-15, -15, -15, -15, -8, -8, -8, -8] * numRepeats * 2
    all_doCTMQC_C = ([True] * 8 + [False] * 8) * numRepeats * 2
    all_doCTMQC_F = ([True] * 8 + [False] * 8)  * numRepeats * 2
    rootFolder = '/scratch/mellis/TullyModelData/CompleteData/Repeat'
    all_nRep = [200]

elif inputs == 'quickFullCTMQC':
    print("Quickly running through all the models with CTMQC (reduced repilcas)")
    numRepeats = 1
    all_velMultiplier = [4, 2, 3, 1, 3, 1.6, 2.5, 1] * numRepeats
    all_maxTime = [2000, 2500, 1300, 5500, 1500, 2500, 2000, 3500] * numRepeats
    all_model = [4, 4, 3, 3, 2, 2, 1, 1] * numRepeats
    all_p_mean = [-15, -15, -15, -15, -8, -8, -8, -8] * numRepeats
    all_doCTMQC_C = ([True] * 8) * numRepeats
    all_doCTMQC_F = ([True] * 8 )  * numRepeats
    rootFolder = '/scratch/mellis/TullyModelData/QuickTests'
    all_nRep = [20]

elif inputs == 'quickFullEhren':
    print("Quickly running through all the models with Ehren (reduced repilcas)")
    numRepeats = 1
    all_velMultiplier = [4, 2, 3, 1, 3, 1.6, 2.5, 1] * numRepeats
    all_maxTime = [2000, 2500, 1300, 5500, 1500, 2500, 2000, 3500] * numRepeats
    all_model = [4, 4, 3, 3, 2, 2, 1, 1] * numRepeats
    all_p_mean = [-15, -15, -15, -15, -8, -8, -8, -8] * numRepeats
    all_doCTMQC_C = ([False] * 8) * numRepeats
    all_doCTMQC_F = ([False] * 8 )  * numRepeats
    rootFolder = '/scratch/mellis/TullyModelData/QuickTests'
    all_nRep = [5]

else:
    print("Carrying out custom input file")
    #numRepeats = 1
    #all_velMultiplier = [4, 2, 3, 1, 3, 1.6, 2.5, 1] * numRepeats
    #all_maxTime = [2000, 2500, 1300, 5500, 1500, 2500, 2000, 3500] * numRepeats
    #all_model = [4, 4, 3, 3, 2, 2, 1, 1] * numRepeats
    #all_p_mean = [-15, -15, -15, -15, -8, -8, -8, -8] * numRepeats
    #all_doCTMQC_C = ([True] * 8) * numRepeats
    #all_doCTMQC_F = ([True] * 8 )  * numRepeats
    #rootFolder = '/scratch/mellis/TullyModelData/ConstSig'
    #nRep = 200
    numRepeats = 1
    all_velMultiplier = [3] * numRepeats
    all_maxTime = [40000] * numRepeats
    all_model = ['lin'] * numRepeats
    all_p_mean = [-15] * numRepeats
    all_doCTMQC_C = [False] * numRepeats
    all_doCTMQC_F = [False]  * numRepeats
    rootFolder = './Data/'
    all_nRep = [1] * numRepeats


s_mean = 0.3
mass = 2000

nSim = min([len(all_velMultiplier), len(all_maxTime),
            len(all_model), len(all_p_mean), len(all_doCTMQC_C),
            len(all_doCTMQC_F), len(all_nRep)])


def setup(pos, vel, coeff, sigma, maxTime, model, doCTMQC_C, doCTMQC_F):
    # All units must be atomic units
    ctmqc_env = {
            'pos': pos,  # Intial Nucl. pos | nrep |in bohr
            'vel': vel,  # Initial Nucl. veloc | nrep |au_v
            'u': coeff,  # Intial WF |nrep, 2| -
            'mass': [mass],  # nuclear mass |nrep| au_m
            'tullyModel': model,  # Which model | | -
            'max_time': maxTime,  # Maximum time to simulate to | | au_t
            'dx': 1e-2,  # The increment for the NACV and grad E calc | | bohr
            'dt': 0.05,  # The timestep | |au_t
            'elec_steps': 1,  # Num elec. timesteps per nucl. one | | -
            'do_QM_F': doCTMQC_F,  # Do the QM force
            'do_QM_C': doCTMQC_C,  # Do the QM force
            'do_sigma_calc': False,  # Dynamically adapt the value of sigma
            'sigma': sigma,  # The value of sigma (width of gaussian)
            'const': 15,  # The constant in the sigma calc
            'nSmoothStep': 7,  # The number of steps to take to smooth the QM intercept
            'gradTol': 1,  # The maximum allowed gradient in Rlk in time.
            'renorm': False,  # Choose whether renormalise the wf
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

        modelStr = "Model_%s" % str(self.ctmqc_env['tullyModel'])
        mom = np.round(self.ctmqc_env['vel'][0] * self.ctmqc_env['mass'][0], 7)
        if int(mom) == mom:
            mom = int(mom)
        momStr = "Kinit_%s" % str(mom).replace(".", "x")
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
    
            mom = np.round(self.ctmqc_env['vel'][0] * self.ctmqc_env['mass'][0], 7)
            momStr = "Kinit_%s" % str(mom).replace(".", "x")
            if int(mom) == mom:
                mom = int(mom)
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
    
            mom = np.round(self.ctmqc_env['vel'][0] * self.ctmqc_env['mass'][0], 7)
            momStr = "Kinit_%s" % str(mom).replace(".", "x")
            if int(mom) == mom:
                mom = int(mom)
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
        
                mom = np.round(self.ctmqc_env['vel'][0] * self.ctmqc_env['mass'][0], 7)
                momStr = "Kinit_%s" % str(mom).replace(".", "x")
                if int(mom) == mom:
                    mom = int(mom)
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
        elif self.ctmqc_env['tullyModel'] == 'lin':
            self.ctmqc_env['Hfunc'] = Ham.create_Hlin
        else:
            print("Tully Model = %s" % str(self.ctmqc_env['tullyModel']))
            msg = "Incorrect tully model chosen. Only 1, 2, 3 and 4 available"
            raise SystemExit(msg)

    def __init_step(self):
        """
        Will carry out the initialisation step (just 1 step without
        wf propagation for RK4)
        """
        nrep = self.ctmqc_env['nrep']
        nstate = self.ctmqc_env['nstate']
        self.ctmqc_env['iSmoothStep'] = -1
        self.ctmqc_env['prevSpike'] = False
        self.ctmqc_env['t'] = 0
        self.ctmqc_env['iter'] = 0

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

        self.__update_vars_step()

    def __calc_quantities(self):
        """
        Will calculate the various paramters to feed into the force and
        electronic propagators. These are then saved in the ctmqc_env dict.
        """
        # Get adiabatic populations
        adPops = np.conjugate(self.ctmqc_env['C']) * self.ctmqc_env['C']
        if np.any(np.abs(adPops.imag) > 0):
            raise SystemExit("Something funny with adiabatic populations")
        self.ctmqc_env['adPops'] = adPops.real
        
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
            except (KeyboardInterrupt, SystemExit):
                print("\n\nOk Exiting Safely")
                return

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

        # Check the norm
        norm = np.sum(self.ctmqc_env['adPops'], axis=1)
        if any(norm > 2):
            print("Error in conserving the norm.")
            print("Norms for all reps [%s]" % (', '.join(norm.astype(str))))
            raise SystemExit("ERROR: Norm Cons")

        # Transform WF
        if self.adiab_diab == 'adiab':
            if self.ctmqc_env['renorm']:
               e_prop.renormalise_all_coeffs(self.ctmqc_env['C'])
            e_prop.trans_adiab_to_diab(self.ctmqc_env)
        else:
            if self.ctmqc_env['renorm']:
               e_prop.renormalise_all_coeffs(self.ctmqc_env['u'])
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

    def __checkS26(self):
        """
        Will check the S26 equation hold if a CTMQC run is being performed
        """
        l, k = 0, 1
        Qlk = self.ctmqc_env['Qlk'][:, l, k]
        fl = self.ctmqc_env['adMom'][:, l]
        fk = self.ctmqc_env['adMom'][:, k]
        Cl = self.ctmqc_env['adPop'][:, l]
        Ck = self.ctmqc_env['adPop'][:, k]
        
        S26 = 2 * Qlk * (fk - fl) * Ck * Cl
        S26 = np.sum(S26, axis=0)
        if S26 > 1e-5:
           print("\nS26 being violated!\n")
           raise SystemExit("S26 Error")


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

        # Save little things like the strs and int vars etc...
        saveTypes = (str, int, float)
        tullyInfo = {i:self.ctmqc_env[i] 
                       for i in self.ctmqc_env
                       if isinstance(self.ctmqc_env[i], saveTypes)}
        np.save("%s/tullyInfo" % self.saveFolder, tullyInfo)
        

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
        if any([all([self.ctmqc_env['do_QM_F'],
                self.ctmqc_env['do_QM_C']]),
               self.ctmqc_env['iter'] < 10]) is False:
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
    nRep = all_nRep[i]
    
    v_mean = 5e-3 * velMultiplier
    v_std = 0  # 2.5e-4 * 0.7
    p_std = 10. / float(v_mean * mass)
    s_std = 0
    
    coeff = [[complex(1, 0), complex(0, 0)]
              for iRep in range(nRep)]
    
    pos = [rd.gauss(p_mean, p_std) for I in range(nRep)]
    vel = [abs(rd.gauss(v_mean, v_std)) for I in range(nRep)]
    
    corrV = 1
    if np.mean(vel) != 0:
        corrV = v_mean / np.mean(vel)
    vel = np.array(vel) * corrV
    
    corrP = 1
    if np.mean(pos) != 0:
        corrP = p_mean / np.mean(pos)
    pos = np.array(pos) * corrP
    pos = np.array([-15])
#    pos = np.array([15 , 15.367833 , 15.686326 , 15.260403 , 15.504686 , 15.933388 
#                    16.337976 , 15.636532 , 15.00771 , 16.897237 , 15.581466 , 15.107695 
#                    14.583034 , 15.559791 , 15.043974 , 15.531766 , 17.474016 , 13.735421 
#                    15.731471 , 13.66913 , 15.313258 , 15.321762 , 14.439246 , 15.879568 
#                    13.978495 , 15.044843 , 14.178503 , 13.628614 , 15.302411 , 14.820269 
#                    15.730734 , 13.509959 , 15.371594 , 14.153199 , 14.873124 , 15.05928 
#                    14.897351 , 18.589138 , 12.137156 , 14.933406 , 14.872558 , 16.886919 
#                    14.147473 , 14.832666 , 14.453683 , 17.6669 , 14.480971 , 14.694849 
#                    14.316093 , 16.694849 , 14.15284 , 12.635001 , 16.338562 , 16.725462 
#                    12.538512 , 15.796553 , 17.417722 , 16.384954 , 16.005783 , 14.129842 
#                    15.68695 , 15.756132 , 14.490892 , 15.646415 , 15.566083 , 14.854284 
#                    13.545996 , 13.145867 , 14.817453 , 16.256434 , 14.746041 , 13.908084 
#                    15.335028 , 15.690294 , 16.788125 , 15.552874 , 14.392097 , 15.433066 
#                    15.11344 , 15.899335 , 13.737726 , 15.185778 , 15.804433 , 15.017083 
#                    15.346517 , 17.367569 , 13.858385 , 16.521751 , 14.695567 , 13.621376 
#                    16.389867 , 16.52056 , 13.621622 , 16.405363 , 13.290941 , 14.205922 
#                    14.167221 , 16.364677 , 14.168563 , 15.493348 , 16.578461 , 14.304925 
#                    15.231736 , 16.092521 , 14.788975 , 16.187365 , 13.491081 , 14.877452 
#                    12.760611 , 14.700896 , 15.469387 , 16.083242 , 18.159379 , 13.176821 
#                    13.831815 , 15.387392 , 14.350278 , 16.229204 , 14.794758 , 15.967629 
#                    15.122151 , 15.474942 , 14.035621 , 14.400129 , 14.953305 , 12.292887 
#                    15.14377 , 14.353471 , 17.667221 , 16.521297 , 17.988397 , 15.275521 
#                    16.027117 , 15.827167 , 13.621698 , 16.234741 , 15.677048 , 15.427303 
#                    12.726123 , 17.256586 , 13.766129 , 17.426225 , 13.96555 , 13.172739 
#                    12.373786 , 15.570675 , 14.03496 , 14.279168 , 14.518897 , 13.904361 
#                    16.460656 , 15.817472 , 13.690729 , 16.288031 , 15.416134 , 14.79421 
#                    15.677633 , 14.739786 , 13.482199 , 16.357893 , 16.819438 , 13.457973 
#                    14.923372 , 14.542594 , 14.811973 , 16.05414 , 12.417571 , 11.992611 
#                    14.696719 , 16.446351 , 14.585188 , 15.57196 , 15.350183 , 13.770343 
#                    14.890207 , 14.827526 , 16.373899 , 13.732227 , 12.650384 , 15.54352 
#                    16.85933 , 14.613893 , 15.844098 , 14.325069 , 15.765524 , 16.419252 
#                    16.603556 , 15.283646 , 15.403568 , 13.934276 , 14.854076 , 15.611304 
#                    15.234211 , 14.913923 , 16.103046 , 15.778487 , 17.110567 , 12.402056 
#                    13.475112 , 15.935844 , 13.420537 , 16.574455 , 16.606278 , 14.363392 
#                    14.718243 , 16.825957 , 13.626913 , 14.268642 , 14.935164 , 15.176821 
#                    15.193979 , 15.027061 , 16.930988 , 12.290582 , 14.18574 , 16.057542 
#                    15.648721 , 14.82847 , 14.562417 , 16.432367 , 15.170263 , 13.149382 
#                    15.628009 , 16.874939 , 14.760119 , 13.239484 , 14.122813 , 13.27244 
#                    13.979364 , 11.561945 , 16.973204 , 17.231339 , 15.995843 , 14.269946 
#                    16.817208 , 13.955989 , 13.435712 , 16.341377 , 16.843078 , 14.502419 
#                    14.453343 , 13.725481 , 16.209513 , 14.231339 , 14.51627 , 15.053177 
#                    14.591009 , 14.936865 , 14.995767 , 17.879682 , 15.882138 , 14.232133 
#                    14.253732 , 15.644431 , 12.938395 , 16.256756 , 15.392532 , 13.976492 
#                    14.911202 , 15.714086 , 16.181961 , 13.167637 , 13.734457 , 13.078272 
#                    13.241638 , 14.769096 , 13.807589 , 13.766034 , 16.562909 , 16.521959 
#                    12.495729 , 15.993972 , 16.334385 , 14.864224 , 15.204241 , 13.275294 
#                    16.420745 , 14.379172 , 16.171624 , 14.132866 , 13.976813 , 14.076174 
#                    18.115575 , 13.22189 , 15.650308 , 15.219453 , 15.872293 , 13.170963 
#                    15.958445 , 13.304547 , 17.887335 , 14.704165 , 15.448071 , 11.046827 
#                    15.126535 , 14.089043 , 16.636494 , 14.758475 , 15.477909 , 15.259118 
#                    16.938301 , 15.364847 , 15.657357 , 14.30791 , 14.456744 , 14.09288 
#                    14.362448 , 16.405533 , 14.781511 , 12.93626 , 15.416588 , 14.052874 
#                    12.752731 , 13.097377 , 14.372879 , 12.970313 , 16.997411 , 14.646831 
#                    16.662629 , 16.102687 , 14.36755 , 15.308421 , 14.943006 , 12.790053 
#                    14.379493 , 15.337069 , 15.511168 , 13.297706 , 14.143996 , 11.523489 
#                    14.675876 , 15.31254 , 15.839752 , 13.676273 , 15.056162 , 12.888091 
#                    14.145508 , 16.320855 , 16.463453 , 14.44569 , 15.144299 , 14.482898 
#                    15.363959 , 15.397747 , 14.435542 , 15.050021 , 14.477739 , 16.000661 
#                    15.677596 , 17.872085 , 14.848728 , 15.827374 , 15.402812 , 15.360501 
#                    14.255017 , 13.870781 , 16.071564 , 15.061113 , 16.119166 , 15.665634 
#                    12.071072 , 13.996466 , 15.63553 , 13.50531 , 13.235704 , 16.189765 
#                    15.440172 , 16.342889 , 14.648135 , 14.009883 , 14.88433 , 13.896992 
#                    15.847519 , 17.242299 , 15.270305 , 13.736763 , 16.653823 , 15.106523 
#                    15.096338 , 14.186005 , 15.987622 , 13.200008 , 13.914075 , 14.242602 
#                    14.155675 , 13.80209 , 14.888507 , 14.029045 , 15.521675 , 13.723591 
#                    15.579198 , 15.057428 , 15.857081 , 14.787294 , 16.80791 , 16.347443 
#                    13.175385 , 14.404154 , 13.624816 , 14.323047 , 13.917854 , 15.090593 
#                    15.470256 , 16.001077 , 15.485563 , 14.091821 , 12.97139 , 15.96899 
#                    13.835273 , 16.035187 , 14.877093 , 13.80124 , 16.210817 , 16.433747 
#                    16.383291 , 13.308137 , 14.693091 , 14.902699 , 17.328962 , 15.408103 
#                    14.395555 , 13.388601 , 13.214105 , 14.565327 , 16.732189 , 14.785366 
#                    13.328773 , 14.942742 , 15.616407 , 15.457954 , 14.224007 , 15.105408 
#                    13.554046 , 13.318663 , 15.812899 , 15.752409 , 14.827545 , 14.386334 
#                    16.623115 , 15.968857 , 16.212045 , 14.707963 , 14.68729 , 15.006633 
#                    14.190427 , 14.850372 , 15.708662 , 13.511773 , 15.991761 , 14.647946 
#                    15.21384 , 14.953003 , 13.550928 , 14.944499 , 14.700726 , 15.525039 
#                    17.546846 , 14.812672 , 15.596451 , 13.473261 , 15.150365 , 15.289637 
#                    14.551438 , 15.211724 , 14.981235 , 14.305605 , 16.581428 , 16.558902 
#                    14.596413 , 15.091595 , 14.184474 , 17.046808 , 13.609056 , 14.890907 
#                    14.267055 , 15.133168 , 15.446559 , 16.68421 , 13.728656 , 16.008145 
#                    15.572093 , 16.788068 , 12.666824 , 18.237859 , 15.277316 , 12.778752 
#                    16.191485 , 14.269549 , 15.335273 , 13.217393 , 15.476076 , 15.910257 
#                    16.041574 , 16.572754 , 16.564024 , 15.821082 , 16.665293 , 15.115084])

    sigma = [rd.gauss(s_mean, s_std) for I in range(nRep)]

    # Now run the simulation
    ctmqc_env = setup(pos, vel, coeff, sigma, maxTime, model,
                      doCTMQC_C, doCTMQC_F)
    return CTMQC(ctmqc_env, rootFolder)


def get_min_procs(nSim, maxProcs):
   """
   This will simply find the maximum amount of processes that can
   be used (e.g nproc = nSim or maxProcs) then minimise this by
   keeping the nsim per proc ratio constant but reducing number
   procs.
   """
   nProc = min([nSim, 16])
   nProc = min([nProc, mp.cpu_count() // 2])
   sims_to_procs = np.ceil(nSim / nProc)
   for i in range(nProc, 1, -1):
       if np.ceil(nSim / i) == sims_to_procs:
           nProc = i
       else:
           break
   return nProc


if nSim > 1:
    import multiprocessing as mp
    
    nProc = get_min_procs(nSim, 16)
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
    #plot.plotDeco(runData)
    #plot.plotRlk_Rl(runData)
    plot.plotNorm(runData)
    plot.plotEcons(runData)
    #plot.plotSigmal(runData)
    #plot.plotQlk(runData)
    #plot.plotS26(runData)
#    plot.plotEpotTime(runData, range(0, runData.ctmqc_env['iter']),
#                      saveFolder='/scratch/mellis/Pics')
    plt.show()

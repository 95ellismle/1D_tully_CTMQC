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
import subprocess

import hamiltonian as Ham
import nucl_prop
import elec_prop as e_prop
import QM_utils as qUt
import plot
#from plottingResults import plotPaperData as plotPaper
inputs = "custom"


#inputs = "FullCTMQC"
#inputs = "FullCTMQCGossel"
#inputs = "FullCTMQCGosselQuick"
#inputs = "FullCTMQCEhren"
#inputs = "quickFullCTMQC"
#inputs = "quickFullEhren"
#inputs = "quickFullEhrenGossel"
#inputs = "MomentumEhren2"
#inputs = "FullEhrenGossel"

rootSaveFold = "/scratch/mellis/TullyModelData"
mfolder_structure = ['ctmqc', 'model', 'mom']

if inputs == "MomentumEhren1":
    mfolder_structure = ['model', 'mom']
    all_velMultiplier = np.arange(0.02, 3, 0.02)
    all_maxTime = [(35.) / (v * 5e-3) for v in all_velMultiplier]
    all_model = [1] * len(all_velMultiplier)
    all_p_mean = [-8] * len(all_velMultiplier)
    all_doCTMQC_C = [False] * len(all_velMultiplier)
    all_doCTMQC_F = [False] * len(all_velMultiplier)
    rootFolder = '%s/MomentumRuns' % rootSaveFold
    all_nRep = [1] * len(all_velMultiplier)

elif inputs == "MomentumEhren2":
    mfolder_structure = ['model', 'mom']
    all_velMultiplier = np.arange(1, 5, 0.02)
    all_maxTime = [(35.) / (v * 5e-3) for v in all_velMultiplier]
    all_model = [2] * len(all_velMultiplier)
    all_p_mean = [-8] * len(all_velMultiplier)
    all_doCTMQC_C = [False] * len(all_velMultiplier)
    all_doCTMQC_F = [False] * len(all_velMultiplier)
    rootFolder = '%s/MomentumRuns' % rootSaveFold
    all_nRep = [1] * len(all_velMultiplier)

elif inputs == "MomentumEhren3":
    mfolder_structure = ['model', 'mom']
    all_velMultiplier = np.arange(1, 4, 0.02)
    all_maxTime = [(35.) / (v * 5e-3) for v in all_velMultiplier]
    all_model = [3] * len(all_velMultiplier)
    all_p_mean = [-15] * len(all_velMultiplier)
    all_doCTMQC_C = [False] * len(all_velMultiplier)
    all_doCTMQC_F = [False] * len(all_velMultiplier)
    rootFolder = '%s/MomentumRuns' % rootSaveFold
    all_nRep = [1] * len(all_velMultiplier)

elif inputs == "MomentumEhren4":
    mfolder_structure = ['model', 'mom']
    all_velMultiplier = np.arange(3, 5, 0.001)
    all_maxTime = [(45.) / (v * 5e-3) for v in all_velMultiplier]
    all_model = [4] * len(all_velMultiplier)
    all_p_mean = [-15] * len(all_velMultiplier)
    all_doCTMQC_C = [False] * len(all_velMultiplier)
    all_doCTMQC_F = [False] * len(all_velMultiplier)
    rootFolder = '%s/MomentumRuns' % rootSaveFold
    all_nRep = [1] * len(all_velMultiplier)

elif inputs == "quickFullEhrenGossel":
    print("Carrying out Ehrenfest simulations!")
    numRepeats = 1
    all_velMultiplier = [4, 3, 3, 2.5,      1,    1,    1.6,  1.5] * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000,  6000, 5000, 2500, 6000] * numRepeats
    all_model = [4, 3, 2, 1,                4,    3,    2,    1] * numRepeats
    all_p_mean = [-20, -15, -8, -15,        -20, -15,  -8,   -15] * numRepeats
    all_doCTMQC_C = ([False] * 8) * numRepeats
    all_doCTMQC_F = ([False] * 8)  * numRepeats
    rootFolder = '/scratch/mellis/TullyModelData/EhrenDataQuick/Repeat'
    all_nRep = [1] * len(all_p_mean)

elif inputs == "FullEhrenGossel":
    print("Carrying out Ehrenfest simulations!")
    numRepeats = 10
    all_velMultiplier = [4, 3, 3, 2.5,      1,    1,    1.6,  1.5] * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000,  6000, 5000, 2500, 6000] * numRepeats
    all_model = [4, 3, 2, 1,                4,    3,    2,    1] * numRepeats
    all_p_mean = [-20, -15, -8, -20,        -20, -15,  -8,   -20] * numRepeats
    all_doCTMQC_C = ([False] * 8) * numRepeats
    all_doCTMQC_F = ([False] * 8)  * numRepeats
    rootFolder = '/scratch/mellis/TullyModelData/EhrenData/Repeat'
    all_nRep = [200] * len(all_p_mean)

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
    all_nRep = [200] * 8 * numRepeats

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
    all_nRep = [200] * 16 * numRepeats

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
    all_nRep = [20] * 8 * numRepeats

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
    all_nRep = [5] * 8 * numRepeats

elif inputs == "FullCTMQCGossel":
    print("Carrying out full Gossel CTMQC testing!")
    numRepeats = 10
    mfolder_structure = ['model', 'mom']
    all_velMultiplier = [4, 3, 3, 2.5,      1,    1,    1.6,  1.5] * numRepeats
    all_maxTime = [2000, 1500, 1500, 4000,  6000, 5000, 2500, 6000] * numRepeats
    all_model = [4, 3, 2, 1,                4,    3,    2,    1] * numRepeats
    all_p_mean = [-20, -15, -8, -20,        -20, -15,  -8,   -20] * numRepeats
    all_doCTMQC_C = ([True] * 8) * numRepeats
    all_doCTMQC_F = ([True] * 8 )  * numRepeats
    rootFolder = '/scratch/mellis/TullyModelData/FullCTMQCGossel_ConstSig/'
    all_nRep = [200] * 8 * numRepeats

elif inputs == "FullCTMQCGosselQuick":
    print("Carrying out quick Gossel CTMQC testing!")
    numRepeats = 1
    all_velMultiplier = [4, 1, 3, 1, 3, 1.6, 2.5, 1.5] * numRepeats
    all_maxTime = [2000, 6000, 1500, 5000, 1500, 2500, 4000, 6000] * numRepeats
    all_model = [4, 4, 3, 3, 2, 2, 1, 1] * numRepeats
    all_p_mean = [-20, -20, -15, -15, -8, -8, -8, -8] * numRepeats
    all_doCTMQC_C = ([True] * 8) * numRepeats
    all_doCTMQC_F = ([True] * 8 )  * numRepeats
    rootFolder = '/scratch/mellis/TullyModelData/FullCTMQCGosselQuick'
    all_nRep = [20] * 8 * numRepeats
else:
    print("Carrying out custom input file")
    numRepeats = 1
    mfolder_structure = []
    all_nRep = [20]  # , 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300] 
    all_velMultiplier = [2.5] * numRepeats * len(all_nRep)
    all_maxTime = [2500] * numRepeats * len(all_nRep)
    all_model = [2] * numRepeats * len(all_nRep)
    all_p_mean = [-15] * numRepeats * len(all_nRep)
    all_doCTMQC_C = ([True] * 2) * numRepeats * len(all_nRep)
    all_doCTMQC_F = ([False] * 2)  * numRepeats * len(all_nRep)
    rootFolder = False #'%s/EhrenData' % rootSaveFold


s_mean = 0.3
mass = 2000

nSim = min([len(all_velMultiplier), len(all_maxTime),
            len(all_model), len(all_p_mean), len(all_doCTMQC_C),
            len(all_doCTMQC_F), len(all_nRep)])
print("Nsim = %i" % nSim)

def setup(pos, vel, coeff, sigma, maxTime, model, doCTMQC_C, doCTMQC_F):
    # All units must be atomic units
    ctmqc_env = {
            'pos': pos,  # Intial Nucl. pos | nrep |in bohr
            'vel': vel,  # Initial Nucl. veloc | nrep |au_v
            'u': coeff,  # Intial WF |nrep, 2| -
            'mass': mass,  # nuclear mass |nrep| au_m
            'tullyModel': model,  # Which model | | -
            'max_time': maxTime,  # Maximum time to simulate to | | au_t
            'dx': 1e-5,  # The increment for the NACV and grad E calc | | bohr
            'dt': 1,  # The timestep | |au_t
            'elec_steps': 2,  # Num elec. timesteps per nucl. one | | -
            'do_QM_F': doCTMQC_F,  # Do the QM force
            'do_QM_C': doCTMQC_C,  # Do the QM force
            'do_sigma_calc': False,  # Dynamically adapt the value of sigma
            'sigma': sigma,  # The value of sigma (width of gaussian)
            'const': 5,  # The constant in the sigma calc
            'nSmoothStep': 7,  # The number of steps to take to smooth the QM intercept
            'gradTol': 0.9,  # The maximum allowed gradient in Rlk in time.
            'renorm': False,  # Choose whether renormalise the wf
            'Qlk_type': 'Min17',  # What method to use to calculate the QM
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

    def __init__(self, ctmqc_env, root_folder = False, folder_structure=['ctmqc', 'model', 'mom']):
        
        # Set everything up
        self.root_folder = root_folder
        self.ctmqc_env = ctmqc_env

        self.__init_tully_model()  # Set the correct Hamiltonian function
        self.__init_nsteps()  # Find how many steps to take
        self.__init_pos_vel_wf()  # set pos vel wf as arrays, get nrep
        self.__init_arrays()  # Create the arrays used
        self.__init_sigma()  # Will initialise the nuclear width
        self.__create_folderpath(folder_structure)

        # Carry out the propagation
        self.__init_step()  # Get things prepared for RK4 (propagate positions)
        self.__main_loop()  # Loop over all steps and propagate
        self.__finalise()  # Finish up and tidy
    
    def __create_folderpath(self, folder_structure):
        """
        Will determine where to store the data. 

        Inputs: folder_sStructure [list <str>] a list of keys to the ctmqc_env
           dictionary that determines the folderstructure of the
           data folders.

        What it does: This function will create the folder that the data
           will be saved in and stored as an attribute named self.save_folder.

        How it works: This is based on the parameters that are inputted into
           the simulation so different runs are stored in different places
           that are easy to find.
        """
        if bool(self.root_folder) is False:
            self.save_folder = False
            return
        self.root_folder = os.path.abspath(self.root_folder)

        # First determine some special params
        CT_str = "Ehren"
        if self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
           CT_str = "CTMQC"
        elif self.ctmqc_env['do_QM_F'] and not self.ctmqc_env['do_QM_C']:
           CT_str = "n-CTMQC"
        elif not self.ctmqc_env['do_QM_F'] and self.ctmqc_env['do_QM_C']:
           CT_str = "e-CTMQC"

        mom = round(self.ctmqc_env['mass'] * self.ctmqc_env['vel'][0], 7)
        if int(mom) == mom: mom = int(mom)
        mom_str = "Kinit=%s" % (str(mom).replace(".", "x").strip())

        model_str = "Model_%i" % self.ctmqc_env['tullyModel']

        params = {i: str(i) + "=" + str(self.ctmqc_env[i]) for i in self.ctmqc_env}
        params['ctmqc'] = CT_str
        params['model'] = model_str
        params['mom'] = mom_str
        
        # Create the folderpath from the structure given
        folderPath = "/".join([params[i] for i in folder_structure])
        self.save_folder = "%s/%s" % (self.root_folder, folderPath)

        # If the folder already exists then move the contents into a Repeat
        #  folder and create more Repeats
        if os.path.isdir(self.save_folder):
            # move any straggling files into a Repeat folder
            oSaveF = self.save_folder[:]
            self.save_folder = "%s/Repeat" % self.save_folder
            if not os.path.isdir(self.save_folder):
               os.mkdir(self.save_folder)
               for f in os.listdir(oSaveF):
                  oldFile = "%s/%s" % (oSaveF, f) 
                  if os.path.isfile(oldFile) and '.npy' in oldFile:
                     os.rename(oldFile, "%s/%s" % (self.save_folder, f))
            
            # Create a new repeat
            count = 1
            oSaveF = self.save_folder[:]
            while os.path.isdir(self.save_folder):
                self.save_folder = oSaveF + "_%i" % count
                count += 1
        
    
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
            self.ctmqc_env['velInit'] = np.mean(self.ctmqc_env['vel'])
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
        self.allEffR = np.zeros((nstep, nstate, nstate))
        if self.ctmqc_env['Qlk_type'] == 'sigmal':
            self.allRl = np.zeros((nstep, nstate))
        elif self.ctmqc_env['Qlk_type'] == 'Min17':
            self.allRl = np.zeros((nstep, nrep))
        else:
            raise SystemExit("Either use `sigmal` or `Min17` for the Qlk_type")
        self.allSigma = np.zeros((nstep, nrep))
        self.allSigmal = np.zeros((nstep, nstate))

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
        self.ctmqc_env['effR'] = np.zeros((nstate, nstate))
        if self.ctmqc_env['Qlk_type'] == 'sigmal':
            self.ctmqc_env['Rl'] = np.zeros(nstate)
        elif self.ctmqc_env['Qlk_type'] == 'Min17':
            self.ctmqc_env['Rl'] = np.zeros(nrep)
        else:
            raise SystemExit("Either use `sigmal` or `Min17` for the Qlk_type")
        self.ctmqc_env['Qlk'] = np.zeros((nrep, nstate, nstate))
        self.ctmqc_env['Qlk_tm'] = np.zeros((nrep, nstate, nstate))
        self.ctmqc_env['Rlk'] = np.zeros((nstate, nstate))
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
                    adMom = 0.0 * self.ctmqc_env['adMom'][irep]
                else:
                    adMom = qUt.calc_ad_mom(self.ctmqc_env, irep, adFrc)
                self.ctmqc_env['adMom'][irep] = adMom

        # Do for all reps
        if self.ctmqc_env['do_QM_F'] or self.ctmqc_env['do_QM_C']:
            #if self.ctmqc_env['do_sigma_calc']:
            #    qUt.calc_sigma(self.ctmqc_env)
            if self.ctmqc_env['Qlk_type'] == 'Min17':
                self.ctmqc_env['Qlk'] = qUt.calc_Qlk_Min17(self.ctmqc_env)
            if self.ctmqc_env['Qlk_type'] == 'sigmal':
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
            self.ctmqc_env['acc'][irep] = Ftot/self.ctmqc_env['mass']

    def __prop_wf(self):
        """
        Will propagate the wavefunction in the correct basis and transform the
        coefficients.
        """
        # Propagate WF
#        t1 = time.time()
        if self.adiab_diab == 'adiab':
            e_prop.do_adiab_prop(self.ctmqc_env)
        else:
            e_prop.do_diab_prop(self.ctmqc_env)
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
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        names = ["pos", "time", "Ftot", "Feh", "Fqm", "E", "C", "u", "|C|^2",
                "H", "f", "Fad", "vel", "Qlk", "Rlk", "RI0", "sigma", "sigmal",
                "NACV", "Rl"]
        arrs = [self.allR, self.allt, self.allF, self.allFeh, self.allFqm,
                self.allE, self.allC, self.allu, self.allAdPop, self.allH,
                self.allAdMom, self.allAdFrc, self.allv, self.allQlk,
                self.allRlk, self.allSigma, self.allSigmal,
                self.allNACV, self.allRl]
        for name, arr in zip(names, arrs):
            savepath = "%s/%s" % (self.save_folder, name)
            np.save(savepath, arr)

        # Save little things like the strs and int vars etc...
        saveTypes = (str, int, float)
        tullyInfo = {i:self.ctmqc_env[i] 
                       for i in self.ctmqc_env
                       if isinstance(self.ctmqc_env[i], saveTypes)}
        np.save("%s/tullyInfo" % self.save_folder, tullyInfo)
        

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
        if self.ctmqc_env['iter'] > 30 and self.save_folder:
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
        print("Finished. Saving in %s" % self.save_folder)
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



def get_norm_drift(runData):
    """
    Will get the norm drift of some run data.
    """
    Ndt = runData.ctmqc_env['dt']
    allNorms = np.sum(runData.allAdPop, axis=2)
    avgNorms = np.mean(allNorms, axis=1)
    fit = np.polyfit(runData.allt, avgNorms, 1)
    return fit[0] * 41341.3745758 * Ndt


def get_ener_drift(runData):
    """
    Will get total energy drift.
    """
    potE = np.sum(runData.allAdPop * runData.allE, axis=2)
    kinE = 0.5 * runData.ctmqc_env['mass'] * (runData.allv**2)
    totE = potE + kinE
    avgTotE = np.mean(totE, axis=1)
    Ndt = runData.ctmqc_env['dt']
    fit = np.polyfit(runData.allt, avgTotE, 1)
    return fit[0] * Ndt * 41341.3745758


def save_vitals(runData):
    normEnerFile = "normEnerDrift.csv"
    firstLine = "Model,CTMQC,NRep,NuclDt,ElecDt,Norm,Ener,GitCommit\n"
    if not os.path.isfile(normEnerFile):
        with open(normEnerFile, "w") as f:
            f.write(firstLine)
    
    with open(normEnerFile, 'a') as f:
        model = runData.ctmqc_env['tullyModel']
        
        CTMQC = "CTMQC"
        if not runData.ctmqc_env['do_QM_C'] and runData.ctmqc_env['do_QM_F']:
            CTMQC = "n-CTMQC"
        elif not runData.ctmqc_env['do_QM_F'] and runData.ctmqc_env['do_QM_C']:
            CTMQC = "e-CTMQC"
        elif not runData.ctmqc_env['do_QM_F'] and not runData.ctmqc_env['do_QM_C']:
            CTMQC = "Ehrenfest"
        
        nrep = runData.ctmqc_env['nrep']
        Ndt = runData.ctmqc_env['dt']
        Edt = Ndt / float(runData.ctmqc_env['elec_steps'])
        norm = get_norm_drift(runData)
        ener = get_ener_drift(runData)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip("\n")
    
        line = (model, CTMQC, nrep, str(Ndt), str(Edt), norm, ener, commit)
        f.write("%i,%s,%i,%s,%s,%.2g,%.2g,%s\n" % line)

    
def doSim(iSim):
    velMultiplier = all_velMultiplier[iSim]
    maxTime = all_maxTime[iSim]
    model = all_model[iSim]
    p_mean = all_p_mean[iSim]
    doCTMQC_C = all_doCTMQC_C[iSim]
    doCTMQC_F = all_doCTMQC_F[iSim]
    nRep = all_nRep[iSim]
    
    v_mean = 5e-3 * velMultiplier
    v_std = 0  # 2.5e-4 * 0.7
    p_std = np.sqrt(2)  # 15. / float(v_mean * mass)
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

    sigma = [rd.gauss(s_mean, s_std) for I in range(nRep)]

    # Now run the simulation
    ctmqc_env = setup(pos, vel, coeff, sigma, maxTime, model,
                      doCTMQC_C, doCTMQC_F)
    runData = CTMQC(ctmqc_env, rootFolder, mfolder_structure)
    save_vitals(runData)
    return runData


def get_min_procs(nSim, maxProcs):
   """
   This will simply find the maximum amount of processes that can
   be used (e.g nproc = nSim or maxProcs) then minimise this by
   keeping the nsim per proc ratio constant but reducing number
   procs.

   N.B. This assumes processes each take the same amount of time.
   """
   nProc = min([nSim, 16])
   nProc = min([nProc, mp.cpu_count() // 2])
   sims_to_procs = np.ceil(nSim / nProc)
   for i in range(nProc+1, 1, -1):
       if np.ceil(nSim / i) == sims_to_procs:
           nProc = i
       else:
           break

   return nProc


if nSim > 1:
    import multiprocessing as mp
    

    nProc = get_min_procs(nSim, 16)
    print("Using %i processes for %i sims" % (nProc, nSim))
    pool = mp.Pool(nProc)
    print("Doing %i sims with %i processes" % (nSim, nProc))
    pool.map(doSim, range(nSim))
else:
    #import test
    for iSim in range(nSim):
        runData = doSim(iSim)

#        test.vel_is_diff_x(runData)

if nSim == 1 and runData.ctmqc_env['iter'] > 50:
#    plotPaper.params['tullyModel'] = runData.ctmqc_env['tullyModel']
#    plotPaper.params['momentum'] = (runData.allv[0, 0] / 0.0005) > 20
#    plotPaper.params['whichSimType'] = ['CTMQC']
#    plotPaper.params['whichQuantity'] = 'pops'
#    f, a = plotPaper.plot_data(plotPaper.params)
    plot.plotPops(runData)
    #plot.plotRabi(runData)
    #plot.plotDeco(runData)
    #plot.plotRlk_Rl(runData)
    #plot.plotNorm(runData)
    #plot.plotEcons(runData)
    #plot.plotSigmal(runData)
    #plot.plotQlk(runData)
    #plot.plotS26(runData)
#    plot.plotEpotTime(runData, range(0, runData.ctmqc_env['iter']),
#                      saveFolder='/scratch/mellis/Pics')
    plt.show()

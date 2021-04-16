from __future__ import print_function
"""
Created on Thu May  9 14:40:43 2019

Remember to check propagator by turning off forces and checking dx/dt

@author: mellis
"""
# This is a horrible way to do this but the python code is only supposed to be used for sandboxing ideas etc..# import input parameters
from input_files import *

import numpy as np
import copy
import matplotlib.pyplot as plt
import random as rd
import datetime
import time
import os
import collections
import re
import subprocess
import json

import hamiltonian as Ham
import nucl_prop
import elec_prop as e_prop
import QM_utils as qUt
import plot


#rootSaveFold = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test"
#rootSaveFold = "/scratch/mellis/TullyModelData/Test"


BASIS = "diab"



def get_time_taken_ordering_dict(all_nrep, all_max_time,
                                 all_dt, all_elec_steps):
    """
    Will return a dictionary with the ordering of simulation times from low to
    high.
    """
    if all_dt is False:
        all_max_time = [i/0.41341373336565040 for i in all_max_time]
    else:
        all_max_time = [float(i)/dt for i, dt in zip(all_max_time, all_dt)]

    if all_elec_steps is False:
        all_elec_steps = [5 for i in range(len(all_nrep))]

    all_estimates = [nrep * max_time * elec_steps
                     for (nrep, max_time, elec_steps) in zip(all_nrep,
                                                             all_max_time,
                                                             all_elec_steps)]
    all_inds = list(range(len(all_estimates)))
    all_sorted_inds = {i: ind[1] for i, ind in enumerate(sorted(zip(all_estimates, all_inds)))}
    return all_sorted_inds



all_lens = [len(all_velMultiplier), len(all_maxTime),
            len(all_model), len(all_p_mean), len(all_doCTMQC_C),
            len(all_doCTMQC_F), len(all_nRep)]
if all_dt is not False: all_lens.append(len(all_dt))
if all_elec_steps is not False: all_lens.append(len(all_elec_steps))
nSim = min(all_lens)
print("Nsim = %i" % nSim)


# A quick dirty function to see how similar 2 floats are
def get_sig_figs_diff(num1, num2):
    min_len = min([len(re.sub("[^0-9]", "", str(num)))
                   for num in (num1, num2)])
    for sig_figs in range(1, 12):
        str1 = ("%."+'%i' % sig_figs +'g') % num1
        str2 = ("%."+'%i' % sig_figs +'g') % num2
        if sig_figs == min_len-1:
             break
        if str1 != str2:
            if sig_figs == 1:
                print("The numbers don't share any common s.f")
                return 0
            print("Same to %i sf" % sig_figs)
            return sig_figs

    print("The numbers are the same")
    return np.inf


def setup(pos, vel, coeff, sigma, maxTime, model, doCTMQC_C, doCTMQC_F,
          dt=0.41341373336565040, elec_steps=5):

    coeffSymb = 'u' if BASIS == 'diab' else 'C'

    print(f"Nuclear Timestep = {dt:.4f} au ({dt*0.02418884254:.2g} fs | {dt*24.18884254:.2g} as)")
    print(f"Electronic Timestep = {dt/elec_steps:.4f} au ({(dt/elec_steps)*0.02418884254:.4f} fs | {(dt/elec_steps)*24.18884254:.2g} as))")
    print(f"Basis: {BASIS}, coeff symbol: {coeffSymb}")


    # All units must be atomic units
    ctmqc_env = {
            'pos': pos,  # Intial Nucl. pos | nrep |in bohr
            'vel': vel,  # Initial Nucl. veloc | nrep |au_v
            coeffSymb: coeff,  # Intial WF |nrep, 2| -
            'mass': mass,  # nuclear mass |nrep| au_m
            'tullyModel': model,  # Which model | | -
            'max_time': maxTime,  # Maximum time to simulate to | | au_t
            'dx': 1e-5,  # The increment for the NACV and grad E calc | | bohr
            'dt': dt,  # The timestep | |au_t
            'elec_steps': elec_steps,  # Num elec. timesteps per nucl. one | | -
            'do_QM_F': doCTMQC_F,  # Do the QM force
            'do_QM_C': doCTMQC_C,  # Do the QM force
            'do_sigma_calc': 'no',  # Dynamically adapt the value of sigma
            'sigma': sigma,  # The value of sigma (width of gaussian)
            'const': 50,  # The constant in the sigma calc
            'nSmoothStep': 5,  # The number of steps to take to smooth the QM intercept
            'gradTol': 0.1,  # The maximum allowed gradient in Rlk in time.
            'renorm': True,  # Choose whether renormalise the wf
            'Qlk_type': 'Min17',  # What method to use to calculate the QM
            'Rlk_smooth': '',  # Apply the smoothing algorithm to Rlk
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

    def __init__(self, ctmqc_env, root_folder = False,
                 folder_structure=['ctmqc', 'model', 'mom'], para=False):

        # Set everything up
        self.root_folder = root_folder
        if root_folder: self.root_folder = os.path.abspath(self.root_folder)
        self.ctmqc_env = ctmqc_env
        self.para = para
        self.save_folder = False
        self.folder_structure = folder_structure

        self.__init_tully_model()  # Set the correct Hamiltonian function
        self.__init_nsteps()  # Find how many steps to take
        self.__init_pos_vel_wf()  # set pos vel wf as arrays, get nrep
        self.__init_arrays()  # Create the arrays used
        self.__init_sigma()  # Will initialise the nuclear width
        if not para:
            self.create_folderpath()

        # Carry out the propagation
        self.__init_step()  # Get things prepared for RK4 (propagate positions)
        self.__main_loop()  # Loop over all steps and propagate
        self.__finalise()  # Finish up and tidy

    def create_folderpath(self):
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

        # First determine some special params
        CT_str = "Ehren"
        if self.ctmqc_env['do_QM_C'] and self.ctmqc_env['do_QM_F']:
           CT_str = "CTMQC"
        elif self.ctmqc_env['do_QM_F'] and not self.ctmqc_env['do_QM_C']:
           CT_str = "n-CTMQC"
        elif not self.ctmqc_env['do_QM_F'] and self.ctmqc_env['do_QM_C']:
           CT_str = "e-CTMQC"

        mom = round(self.ctmqc_env['mass'] * self.ctmqc_env['velInit'], 7)
        if int(mom) == mom: mom = int(mom)
        mom_str = "Kinit_%s" % (str(mom).replace(".", "x").strip())

        model_str = f"Model_{self.ctmqc_env['tullyModel']}"

        params = {i: str(i) + "=" + str(self.ctmqc_env[i]) for i in self.ctmqc_env}
        params['ctmqc'] = CT_str
        params['model'] = model_str
        params['mom'] = mom_str
        params['sigma'] = "Sig_%.3g" % np.mean(self.ctmqc_env['sigma'])

        # Create the folderpath from the structure given
        folderPath = "/".join([params[i] for i in self.folder_structure])
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
            pos_std = np.std(self.ctmqc_env['pos'])
            if pos_std == 0 and v_std == 0:
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
            if nrep != nrep1: changes = True
            nrep = np.min([nrep1, nrep])
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

            if nrep != nrep1: changes = True
            nrep = np.min([nrep1, nrep])
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
        nstate, nstep = self.ctmqc_env['nstate'], self.ctmqc_env['nsteps'] + 1
        if 'mass' in self.ctmqc_env:
            self.ctmqc_env['mass'] = np.array(self.ctmqc_env['mass'])
        else:
            raise SystemExit("Mass not specified in startup")
        self.saveIter = 0
        self.tenSteps = False
        if nstep > 10000:
            nstep = int(np.ceil(nstep / 10)) + 1
            self.tenSteps = True

        if 'extrapolation' in self.ctmqc_env['Rlk_smooth']:
            nums = re.findall("[0-9]", self.ctmqc_env['Rlk_smooth'])
            self.ctmqc_env['polynomial_order'] = 4
            if len(nums) > 1:
                self.ctmqc_env['polynomial_order'] = int(''.join(nums))

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
        self.allAlpha = np.zeros((nstep, nrep))
        self.allRlk = np.zeros((nstep, nstate, nstate))
        self.allEffR = np.zeros((nstep, nstate, nstate, nrep))
        self.allIsSpiking = np.zeros(nstep, dtype=bool)
        self.allClusters = []
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
        self.ctmqc_env['clusters'] = {}
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
        self.ctmqc_env['effR'] = np.zeros((nstate, nstate, nrep))
        if self.ctmqc_env['Qlk_type'] == 'sigmal':
            self.ctmqc_env['altR'] = np.zeros(nstate)
        elif self.ctmqc_env['Qlk_type'] == 'Min17':
            self.ctmqc_env['altR'] = np.zeros(nrep)
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
        options = {1: Ham.create_H1, 2: Ham.create_H2,
                   3: Ham.create_H3, 4: Ham.create_H4,
                   'lin': Ham.create_Hlin, 'big': Ham.createBigHam,
                   'mult1': Ham.createManyCross,
                   'constHigh': Ham.constantHighCouplings,
                  }
        if self.ctmqc_env['tullyModel'] in options:
            self.ctmqc_env['Hfunc'] = options[self.ctmqc_env['tullyModel']]
        else:
            print("Tully Model = %s" % str(self.ctmqc_env['tullyModel']))
            msg = "Incorrect tully model chosen. Choose from:\n\t*" % \
                    ('\n\t* '.join(options.keys()))
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
        self.ctmqc_env['extrapCount'] = 0
        self.ctmqc_env['spike_region_count'] = 0
        self.ctmqc_env['poss_spike'] = False
        self.ctmqc_env['threshold'] = 9999
        if self.ctmqc_env['Rlk_smooth'] == "RI0":
            self.ctmqc_env['nSmoothStep'] = 0
        self.allTimes = {'step': [], 'force': [], 'wf_prop': [],
                         'transform': [], 'calcQM':[], 'prep': [],
                         "get pops": [],}

        # Calculate the Hamiltonian
        for irep in range(nrep):
            pos = self.ctmqc_env['pos'][irep]
            self.ctmqc_env['H'][irep] = self.ctmqc_env['Hfunc'](pos)
            E, U = np.linalg.eigh(self.ctmqc_env['H'][irep])
            self.ctmqc_env['E'][irep], self.ctmqc_env['U'][irep] = E, U


        # Transform the coefficieints
        if 'u' in self.ctmqc_env:
            #self.ctmqc_env['C'] = np.zeros((nrep, nstate),
            #                               dtype=complex)
            #e_prop.trans_diab_to_adiab(self.ctmqc_env)

            # A quick hack to simulate adiabatic ground state initialisation
            # using the diabatic propagation.
            self.ctmqc_env['C'] = np.array([[complex(1, 0), complex(0, 0)]
                                   for iRep in range(nrep)])
            self.ctmqc_env['u'] = np.zeros((nrep, nstate),
                                           dtype=complex)
            e_prop.trans_adiab_to_diab(self.ctmqc_env)
        else:
            self.ctmqc_env['u'] = np.zeros((nrep, nstate),
                                           dtype=complex)
            e_prop.trans_adiab_to_diab(self.ctmqc_env)

        # Calculate the QM, adMom, adPop, adFrc.
        if self.ctmqc_env['do_sigma_calc'].lower() == 'no':
           self.ctmqc_env['alpha'][:] = 1 / (2 * (self.ctmqc_env['sigma']**2))
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
                if all(Ck > self.ctmqc_env['threshold']
                        for Ck in self.ctmqc_env['adPops'][irep]):
                    self.ctmqc_env['adMom'][:, :] =  0.0

                doQM = True
                adMom = qUt.calc_ad_mom(self.ctmqc_env, irep, adFrc)
                self.ctmqc_env['adMom'][irep] = adMom

        # Do for all reps
        t1 = time.time()
        if self.ctmqc_env['do_QM_F'] or self.ctmqc_env['do_QM_C']: # and doQM:
            #if self.ctmqc_env['do_sigma_calc']:
            #    qUt.calc_sigma(self.ctmqc_env)
            if self.ctmqc_env['Qlk_type'] == 'Min17':
                self.ctmqc_env['Qlk'] = qUt.calc_Qlk_Min17_opt(self)
            if self.ctmqc_env['Qlk_type'] == 'sigmal':
                self.ctmqc_env['Qlk'] = qUt.calc_Qlk_2state(self.ctmqc_env)
        self.allTimes['calcQM'].append(time.time() - t1)
#        print("\n")

    def __main_loop(self):
        """
        Will loop over all steps and propagate the dynamics
        """
        nstep = self.ctmqc_env['nsteps']

        for istep in range(nstep):
            try:
                t1 = time.time()
                if self.tenSteps and self.ctmqc_env['iter'] % 10 == 0:
                   self.__save_data()
                elif self.tenSteps is False:
                   self.__save_data()
                self.__ctmqc_step()
                self.ctmqc_env['t'] += self.ctmqc_env['dt']
                self.ctmqc_env['iter'] += 1

                t2 = time.time()

                # Print some useful info (if not doing parallel sims)
                if not self.para or self.ctmqc_env['iter'] % 100 == 0:
                    self.allTimes['step'].append(t2 - t1)
                    avgTime = np.mean(self.allTimes['step'])
                    msg = "\rStep %i/%i  Time Taken = %.2gs" % (istep, nstep,
                                                                avgTime)
                    timeLeft = int((nstep - istep) * avgTime)
                    timeLeft = str(datetime.timedelta(seconds=timeLeft))
                    msg += "  Time Left = %s" % (timeLeft)
                    percentComplete = (float(istep) / float(nstep)) * 100
                    msg += "  %i%% Complete" % (percentComplete)
        #            print(" "*200, end="\r")
                    print(msg,
                          end="\r")

            except (KeyboardInterrupt, SystemExit) as E:
                print("\n\n\n\n\n\n\n\n\n------------\n\n\n\n\n\n\n\n\n\n\n\n")
                print("\n\nOk Exiting Safely")
                print("\n\n\n\n\n\n\n\n\n------------\n\n\n\n\n\n\n\n\n\n\n\n")
                print(E)
                return
        if self.tenSteps and self.ctmqc_env['iter'] % 10 == 0:
           self.__save_data()
        elif self.tenSteps is False:
           self.__save_data()

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
        t1 = time.time()
        if self.adiab_diab == 'adiab':
            e_prop.do_adiab_prop(self.ctmqc_env)
        else:
            e_prop.do_diab_prop(self.ctmqc_env)
        t2 = time.time()

        # Check the norm
        norm = np.sum(self.ctmqc_env['adPops'], axis=1)
        if any(norm > 30):
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

        # Get adiabatic populations
        adPops = np.conjugate(self.ctmqc_env['C']) * self.ctmqc_env['C']
        if np.any(np.abs(adPops.imag) > 0):
            raise SystemExit("Something funny with adiabatic populations")
        self.ctmqc_env['adPops'] = adPops.real
        t4 = time.time()

        self.allTimes['wf_prop'].append(t2 - t1)
        self.allTimes['transform'].append(t3 - t2)
        self.allTimes['get pops'].append(t4 - t3)

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
        istep = self.saveIter

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
        self.allRl[istep] = self.ctmqc_env['altR']
        self.allAlphal[istep] = self.ctmqc_env['alphal']
        self.allAlpha[istep] = self.ctmqc_env['alpha']
        self.allClusters.append(self.ctmqc_env['clusters'])

        self.allt[istep] = self.ctmqc_env['t']

        self.saveIter += 1

    def __chop_arrays(self):
        """
        Will splice the arrays to the appropriate size (to num steps done)
        """
        self.ctmqc_env['iter'] -= 1
        self.allR = self.allR[:self.saveIter]
        self.allt = self.allt[:self.saveIter]
        self.allNACV = self.allNACV[:self.saveIter]
        self.allF = self.allF[:self.saveIter]
        self.allFeh = self.allFeh[:self.saveIter]
        self.allFqm = self.allFqm[:self.saveIter]
        self.allE = self.allE[:self.saveIter]
        self.allC = self.allC[:self.saveIter]
        self.allu = self.allu[:self.saveIter]
        self.allAdPop = self.allAdPop[:self.saveIter]
        self.allH = self.allH[:self.saveIter]
        self.allAdMom = self.allAdMom[:self.saveIter]
        self.allAdFrc = self.allAdFrc[:self.saveIter]
        self.allv = self.allv[:self.saveIter]
        self.allQlk = self.allQlk[:self.saveIter]
        self.allRlk = self.allRlk[:self.saveIter]
        self.allEffR = self.allEffR[:self.saveIter]
        self.allSigma = self.allSigma[:self.saveIter]
        self.allSigmal = self.allSigmal[:self.saveIter]
        self.allRl = self.allRl[:self.saveIter]
        self.allAlphal = self.allAlphal[:self.saveIter]
        self.allAlpha = self.allAlpha[:self.saveIter]

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



    def store_data(self):
        """
        Will save all the arrays to disc as numpy binary files.
        """
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        names = ["pos", "time", "Ftot", "Feh", "Fqm", "E", "C", "u", "|C|^2",
                "H", "f", "Fad", "vel", "Qlk", "Rlk", "sigma", "sigmal",
                "NACV", "RI0", "effR", "alpha"]
        arrs = [self.allR, self.allt, self.allF, self.allFeh, self.allFqm,
                self.allE, self.allC, self.allu, self.allAdPop, self.allH,
                self.allAdMom, self.allAdFrc, self.allv, self.allQlk,
                self.allRlk, self.allSigma, self.allSigmal,
                self.allNACV, self.allRl, self.allEffR, self.allAlpha]
        for name, arr in zip(names, arrs):
            savepath = "%s/%s" % (self.save_folder, name)
            np.save(savepath, arr)

        # Save little things like the strs and int vars etc...
        saveTypes = (str, int, float)
        tullyInfo = {i:self.ctmqc_env[i]
                       for i in self.ctmqc_env
                       if isinstance(self.ctmqc_env[i], saveTypes)}
        tullyInfo['propagation_basis'] = BASIS

        #np.save("%s/tullyInfo" % self.save_folder, tullyInfo)
        with open("%s/tullyInfo.json" % self.save_folder, 'w') as f:
            json.dump(tullyInfo, f)


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
        if self.save_folder and not self.para:
            self.store_data()

        # Run tests on data (only after Ehrenfest, CTMQC normally fails!)
        if (self.ctmqc_env['do_QM_F'] or self.ctmqc_env['iter'] < 10) is False:
            if self.tenSteps is False:
               self.__checkVV()

        # Print some useful info
        if not self.para:
            sumTime = np.sum(self.allTimes['step'])
            nstep = self.ctmqc_env['iter']
            msg = "\r                                                             "
            msg += "                                                              "
            msg += "\n\n***\n"
            timeTaken = np.ceil(sumTime)
            timeTaken = str(datetime.timedelta(seconds=timeTaken))
            msg += "Steps = %i   Total Time Taken__prop_wf = %ss" % (nstep, timeTaken)
            msg += "  Avg. Time Per Step = %.2gs" % np.mean(self.allTimes['step'])
            msg += "  All Done!\n***\n"

            msg += "\n\nAverage Times:"
            print(msg)
        if self.save_folder is not False:
            print("Finished. Saving in %s" % self.save_folder)
        if not self.para:
            print_timings(self.allTimes, 1)

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

        line = (str(model), CTMQC, nrep, str(Ndt), str(Edt), norm, ener, commit)
        f.write("%s,%s,%i,%s,%s,%.2g,%.2g,%s\n" % line)


def doSim(iSim, para=False):
    velMultiplier = all_velMultiplier[iSim]
    maxTime = all_maxTime[iSim]
    model = all_model[iSim]
    p_mean = all_p_mean[iSim]
    doCTMQC_C = all_doCTMQC_C[iSim]
    doCTMQC_F = all_doCTMQC_F[iSim]
    nRep = all_nRep[iSim]

    v_mean = 5e-3 * velMultiplier
    pos_std = np.sqrt(2)  # 15. / float(v_mean * mass)
    v_std = 1. /(2000. * pos_std)
    s_std = 0


    coeff = [[complex(1, 0), complex(0, 0)]
              for iRep in range(nRep)]
    if BASIS == 'diab':
        if model == 3 or model == 4:
            coeff = [[complex(0, 0), complex(1, 0)]
                        for iRep in range(nRep)]


    rd.seed(1)
    pos = [rd.gauss(p_mean, pos_std) for I in range(nRep)]
    vel = [abs(rd.gauss(v_mean, v_std)) for I in range(nRep)]


    corrV = 1
    if np.mean(vel) != 0:
        corrV = v_mean / np.mean(vel)
    vel = np.array(vel) * corrV

    corrP = 1
    if np.mean(pos) != 0:
        corrP = p_mean / np.mean(pos)
    pos = np.array(pos) * corrP

    s_min = all_sig_min[iSim]
    sigma = [rd.gauss(s_min, s_std) for I in range(nRep)]

    elec_steps = 5
    if all_elec_steps:
        elec_steps = all_elec_steps[iSim]

    dt = 0.41341373336565040
    if all_dt:
        dt = all_dt[iSim]

    # Now run the simulation
    ctmqc_env = setup(pos, vel, coeff, sigma, maxTime, model,
                      doCTMQC_C, doCTMQC_F, dt, elec_steps)
    runData = CTMQC(ctmqc_env, rootFolder, mfolder_structure, para)
    save_vitals(runData)
    return runData


def get_nproc(nSim, maxProcs):
   """
   This will simply find the maximum amount of processes that can
   be used (e.g nproc = nSim or maxProcs) then minimise this by
   keeping the nsim per proc ratio constant but reducing number
   procs.

   N.B. This assumes processes each take the same amount of time,
        which holds due to the way the work is divided into
        blocks.
   """
   if nSim <= maxProcs:
       return nSim

   nProc = min([nSim, maxProcs])
   nProc = min([nProc, mp.cpu_count()])
   sims_to_procs = np.ceil(nSim / nProc)
   for i in range(nProc+1, 1, -1):
       if np.ceil(nSim / i) == sims_to_procs:
           nProc = i
       else:
           break

   if nProc > 1 and nSim % (nProc-1) == 0:
       nProc -= 1

   return nProc


def para_doSim(iSim):
    """
    A wrapper for the do_sim function in order to pass more args from map to it
    """
    print("Starting Simulation %i" % iSim)
    runData = doSim(iSim, para=True)
    print("Completed Simulation %i\n" % iSim)
    return runData

if nSim > 1 and do_parallel:
    import multiprocessing as mp


    nProc = get_nproc(nSim, 12)
    print("Using %i processes" % (nProc))
    pool = mp.Pool(nProc)
#    print("Doing %i sims with %i processes" % (nSim, nProc))

    # This divides the runs into blocks of nProc simulations. Once the block is
    #  complete then the data is saved and the code moves onto the next block.
    # This means that often all the data isn't lost if the code crashes in an
    #  unsafe way.
    order_dict = get_time_taken_ordering_dict(all_nRep, all_maxTime,
                                              all_dt, all_elec_steps)

    all_SimSets = []
    newArr = []
    count = 0
    while count < len(order_dict):
        if count % nProc == 0:
            if len(newArr) > 0:
               all_SimSets.append(newArr)
               newArr = []
        newArr.append(order_dict[count])
        count += 1
    all_SimSets.append(newArr)

    print("Doing %i simulations" % count)
    for simulation_set in all_SimSets:
        allRunData = pool.map(para_doSim, simulation_set)
        print("\n\n\nCompleted all procs, writing data\n\n\n")
        for runData in allRunData:
            runData.create_folderpath()
            runData.store_data()
else:
    #import test
    for iSim in range(nSim):
        runData = doSim(iSim)

#        test.vel_is_diff_x(runData)

if nSim == 1 and runData.ctmqc_env['iter'] > 50:
#    plot.plotRlk_gradRlk(runData)
##    plotPaper.params['tullyModel'] = runData.ctmqc_env['tullyModel']
##    plotPaper.params['momentum'] = (runData.allv[0, 0] / 0.0005) > 20
##    plotPaper.params['whichSimType'] = ['CTMQC']
##    plotPaper.params['whichQuantity'] = 'pops'
##    f, a = plotPaper.plot_data(plotPaper.params)
    plot.plotPops(runData)
    #plot.plotPos(runData)
    #plot.plotNACV(runData)
#    plot.plotNorm(runData)
#    #plot.plotRabi(runData)
#    plot.plotDeco(runData)
#    plot.plotSigma(runData)
    #plot.plotRlk_Rl(runData)
    plot.plotNorm(runData)
#    plot.plotEcons(runData)
#    #plot.plotSigmal(runData)
    #plot.plotQlk(runData)
#    #plot.plotS26(runData)
##    plot.plotEpotTime(runData, range(0, runData.ctmqc_env['iter']),
##                      saveFolder='/scratch/mellis/Pics')
    pass



plt.show()

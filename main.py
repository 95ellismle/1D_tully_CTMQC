from __future__ import print_function
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
import collections
import os
import datetime as dt
import time

import hamiltonian as Ham
import nucl_prop
import elec_prop as e_prop
#import plot
import QM_utils as qUt


all_nReps = [1]# * 6 * 5
all_vel_mults = [2.5]#, 1, 3, 1.6, 2.5, 1] * 5
all_p_means = [-8]#, -15, -8, -8, -8, -8] * 5
all_models = [1]#, 3, 2, 2, 1, 1] * 5
all_max_times = [2500]#, 5500, 1500, 2500, 2000, 3500] * 5

all_v_means = [5e-3 * i for i in all_vel_mults]
nSim = np.min([len(i) for i in (all_nReps, all_v_means,
                                all_p_means, all_models,
                                all_max_times)])

savePath = False # "/temp/mellis/TullyModels/SmallerPosRepeats/Repeat"


dT = 0.5
elecSteps = 1
doCTMQC = False

doCTMQC_C = doCTMQC
doCTMQC_F = doCTMQC

#elecProp = e_prop.elecProp(ctmqc_env)
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


class main(object):
    """
    Will carry out the full propagation from intialisation to end.
    """
    allR = []
    allt = []

    def __init__(self, ctmqc_env, root_folder = False):
        self.ctmqc_env = ctmqc_env
        self.root_folder = root_folder
        
        self.__create_folderpath()
        self.__init_tully_model()
        self.__init_nsteps()
        self.__init_pos_vel_wf()
        self.__init_arrays()

        self.__init_step()
        self.__main_loop()
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
#        if self.ctmqc_env['do_sigma_calc']:
#            sigStr = "varSig"
#        else:
        sigStr = "%.2gSig" % self.ctmqc_env['sigma'][0]
        self.saveFolder = "%s/%s/%s/%s/%s" % (self.root_folder, eHStr,
                                              modelStr, momStr, sigStr)
        
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
#            if self.ctmqc_env['do_sigma_calc']:
#                sigStr = "varSig"
#            else:
            sigStr = "%.2gSig" % self.ctmqc_env['sigma'][0]
            self.saveFolder = "%s/%s/%s/%s/%s" % (self.root_folder, eHStr,
                                                  modelStr, momStr, sigStr)
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
                sigStr = "%.2gSig" % self.ctmqc_env['sigma'][0][0]
            self.saveFolder = "%s/%s/%s/%s/%s" % (self.root_folder, eHStr,
                                                  modelStr, momStr, sigStr)
            
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
                mom = np.round(self.ctmqc_env['vel'][0][0] * self.ctmqc_env['mass'][0])
                momStr = "Kinit_%i" % int(mom)
                if self.ctmqc_env['do_sigma_calc']:
                    sigStr = "varSig"
                else:
                    sigStr = "%.2gSig" % self.ctmqc_env['sigma'][0][0]
                self.saveFolder = "%s/%s/%s/%s/%s" % (self.root_folder, eHStr,
                                                      modelStr, momStr, sigStr)
                count += 1
        print("\n%s" % self.saveFolder, end="\n")

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
        self.allR = np.zeros((nstep, nrep))
        self.allF = np.zeros((nstep, nrep))
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
        self.allNACV = np.zeros((nstep, nrep, nstate, nstate), dtype=complex)
        self.allRlk = np.zeros((nstep, nstate, nstate))
        self.allAlpha = np.zeros((nstep, nrep))
        self.allRI0 = np.zeros((nstep, nrep))
        self.allSigma = np.zeros((nstep, nrep))

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
        self.ctmqc_env['Qlk'] = np.zeros((nrep, nstate, nstate))
        self.ctmqc_env['Qlk_tm'] = np.zeros((nrep, nstate, nstate))
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
        else:
            print("Tully Model = %i" % self.ctmqc_env['tullyModel'])
            msg = "Incorrect tully model chosen. Only 1, 2 and 3 available"
            raise SystemExit(msg)

    def __update_vars_step(self):
        """
        Will update the time-dependant variables in the ctmqc environment.
        """
        self.ctmqc_env['pos_tm'] = copy.deepcopy(self.ctmqc_env['pos'])
        self.ctmqc_env['H_tm'] = copy.deepcopy(self.ctmqc_env['H'])
        self.ctmqc_env['U_tm'] = copy.deepcopy(self.ctmqc_env['U'])
        if self.adiab_diab == 'adiab':
            self.ctmqc_env['vel_tm'] = copy.deepcopy(self.ctmqc_env['vel'])
            self.ctmqc_env['NACV_tm'] = copy.deepcopy(self.ctmqc_env['NACV'])
        if ctmqc_env['do_QM_C'] or ctmqc_env['do_QM_F']:
            self.ctmqc_env['Qlk_tm'] = copy.deepcopy(self.ctmqc_env['Qlk'])
            self.ctmqc_env['adMom_tm'] = copy.deepcopy(self.ctmqc_env['adMom'])

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
                                         self.ctmqc_env['adPops'][irep],
                                         Qlk,
                                         self.ctmqc_env['adMom'][irep],
                                         self.ctmqc_env)

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
        if self.ctmqc_env['do_QM_C']:
#            if self.adiab_diab == 'adiab':
#                e_prop.do_adiab_prop_QM(self.ctmqc_env, self.allTimes)
#            else:
            e_prop.do_diab_prop_QM(self.ctmqc_env)
        else:
#            if self.adiab_diab == 'adiab':
#                e_prop.do_adiab_prop_ehren(self.ctmqc_env)
#            else:
            e_prop.do_diab_prop_ehren(self.ctmqc_env)

        # Transform WF
#        if self.adiab_diab == 'adiab':
##            if self.ctmqc_env['iter'] % 30 == 0:
##                self.ctmqc_env['C'] = e_prop.renormalise_all_coeffs(
##                                                           self.ctmqc_env['C'])
#            for irep in range(ctmqc_env['nrep']):
#                u = np.matmul(np.array(ctmqc_env['U'][irep]),
#                              np.array(ctmqc_env['C'][irep]))
#                self.ctmqc_env['u'][irep] = u
#        else:
#            if self.ctmqc_env['iter'] % 30 == 0:
#                self.ctmqc_env['u'] = e_prop.renormalise_all_coeffs(
#                                                   self.ctmqc_env['u'])
        for irep in range(ctmqc_env['nrep']):
            C = np.matmul(np.array(ctmqc_env['U'][irep].T),
                          np.array(ctmqc_env['u'][irep]))
            self.ctmqc_env['C'][irep] = C

    def __calc_quantities(self):
        """
        Will calculate the various paramters to feed into the force and
        electronic propagators. These are then saved in the ctmqc_env dict.
        """
        # Do for each rep
        for irep in range(self.ctmqc_env['nrep']):
            # Get Hamiltonian
            pos = self.ctmqc_env['pos'][irep]
            self.ctmqc_env['H'][irep] = self.ctmqc_env['Hfunc'](pos)
            E, U = np.linalg.eigh(self.ctmqc_env['H'][irep])
            self.ctmqc_env['E'][irep] = E
            self.ctmqc_env['U'][irep] = U

            # Get adiabatic forces
            adFrc = qUt.calc_ad_frc(pos, self.ctmqc_env)
            self.ctmqc_env['adFrc'][irep] = adFrc
            # Get adiabatic populations
            pop = e_prop.calc_ad_pops(self.ctmqc_env['C'][irep],
                                      self.ctmqc_env)
            self.ctmqc_env['adPops'][irep] = pop
            # Get adiabatic NACV
            self.ctmqc_env['NACV'][irep] = Ham.calcNACV(irep,
                                                           self.ctmqc_env)
            # Get the QM quantities
            if self.ctmqc_env['do_QM_F'] or self.ctmqc_env['do_QM_C']:
                if any(Ck > 0.995 for Ck in pop):
                    adMom = 0.0
                else:
                    adMom = qUt.calc_ad_mom(self.ctmqc_env, irep,
                                            adFrc)
                self.ctmqc_env['adMom'][irep] = adMom
        # Do for all reps
        if self.ctmqc_env['do_QM_F'] or self.ctmqc_env['do_QM_C']:
#            if self.ctmqc_env['do_sigma_calc']:
#                qUt.calc_sigma(self.ctmqc_env)
            self.ctmqc_env['Qlk'] = np.zeros((self.ctmqc_env['nrep'],
                                              self.ctmqc_env['nstate'],
                                              self.ctmqc_env['nstate']))
            self.ctmqc_env['Qlk'] = qUt.calc_Qlk(self.ctmqc_env)
#             = QM
#            self.ctmqc_env['Qlk'][:, 1, 0] = QM

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
            _, self.ctmqc_env['U'][irep] = np.linalg.eigh(self.ctmqc_env['H'][irep])
        # Transform the coefficieints
        # Transform WF
        if self.adiab_diab == 'adiab':
            self.ctmqc_env['u'] = np.zeros_like(ctmqc_env['C'])
            self.ctmqc_env['C'] = e_prop.renormalise_all_coeffs(
                                                           self.ctmqc_env['C'])
            for irep in range(ctmqc_env['nrep']):
                u = np.matmul(np.array(ctmqc_env['U'][irep]),
                              np.array(ctmqc_env['C'][irep]))
                self.ctmqc_env['u'][irep] = u
        else:
            self.ctmqc_env['C'] = np.zeros_like(ctmqc_env['u'])
            self.ctmqc_env['u'] = e_prop.renormalise_all_coeffs(
                                                           self.ctmqc_env['u'])
            for irep in range(ctmqc_env['nrep']):
                C = np.matmul(np.array(ctmqc_env['U'][irep].T),
                              np.array(ctmqc_env['u'][irep]))
                self.ctmqc_env['C'][irep] = C

        self.__calc_quantities()
        self.__calc_F()

        self.ctmqc_env['t'] = 0
        self.ctmqc_env['iter'] = 0
        self.__update_vars_step()

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

    def __ctmqc_step(self):
        """
        Will carry out a single step in the CTMQC.
        """
        dt = self.ctmqc_env['dt']

        self.ctmqc_env['vel'] += 0.5 * self.ctmqc_env['acc'] * dt  # half dt
        self.ctmqc_env['pos'] += self.ctmqc_env['vel']*dt  # full dt


        self.__calc_quantities()
        self.__prop_wf()
        self.__calc_F()

        self.ctmqc_env['vel'] += 0.5 * self.ctmqc_env['acc'] * dt  # full dt

        self.__update_vars_step()  # Save old data

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
        self.allQlk[istep] = self.ctmqc_env['Qlk']
        self.allRlk[istep] = self.ctmqc_env['Rlk']
        self.allRI0[istep] = self.ctmqc_env['RI0']
        self.allAlpha[istep] = self.ctmqc_env['alpha']
        self.allNACV[istep] = self.ctmqc_env['NACV']
        self.allSigma[istep] = self.ctmqc_env['sigma']

        self.allt[istep] = self.ctmqc_env['t']

    def __chop_arrays(self):
        """
        Will splice the arrays to the appropriate size (to num steps done)
        """
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
        self.allRlk = self.allRlk[:self.ctmqc_env['iter']]
        self.allNACV = self.allNACV[:self.ctmqc_env['iter']]
        self.allRI0 = self.allRI0[:self.ctmqc_env['iter']]
        self.allAlpha = self.allAlpha[:self.ctmqc_env['iter']]
        self.allSigma = self.allSigma[:self.ctmqc_env['iter']]

    def __store_data(self):
        """
        Will save all the arrays as numpy binary files.
        """
        if not os.path.isdir(self.saveFolder):
            os.makedirs(self.saveFolder)

        names = ["pos", "time", "Ftot", "Feh", "Fqm", "E", "C", "u", "|C|^2",
                "H", "f", "Fad", "vel", "Qlk", "Rlk", "RI0", "sigma", "alpha",]
        arrs = [self.allR, self.allt, self.allF, self.allFeh, self.allFqm,
                self.allE, self.allC, self.allu, self.allAdPop, self.allH,
                self.allAdMom, self.allAdFrc, self.allv, self.allQlk,
                self.allRlk, self.allRI0, self.allSigma, self.allAlpha]
        for name, arr in zip(names, arrs):
            savepath = "%s/%s" % (self.saveFolder, name)
            np.save(savepath, arr)

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
        print("Finished. Saving in %s" % self.saveFolder)
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

for isim in range(nSim):
    p_std = 10 / (2000 * all_v_means[isim])
    v_std = 0#5e-4
    pos = [rd.gauss(all_p_means[isim], p_std) for i in range(all_nReps[isim])]
#    pos = [
#        -16.054, -14.035, -14.369, -15.661, -14.884, -15.211, -14.389, -14.506, 
#        -14.591, -14.688, -14.45, -15.151, -14.848, -15.097, -14.304, -15.213, 
#        -15.925, -15.778, -14.472, -15.466, -14.947, -15.643, -15.239, -16.611, 
#        -15.958, -14.28, -15.197, -14.83, -14.809, -14.574, -13.126, -15.65, 
#        -14.712, -14.167, -14.71, -14.104, -14.478, -16.195, -15.313, -14.382, 
#        -16.042, -15.603, -15.293, -15.732, -14.927, -13.336, -15.039, -14.262, 
#        -14.18, -15.077, -16.097, -15.253, -14.935, -15.292, -14.763, -15.687, 
#        -15.21, -14.621, -14.574, -14.832, -16.41, -15.499, -14.484, -15.391, 
#        -14.113, -13.065, -16.928, -14.155, -15.665, -14.256, -14.989, -14.636, 
#        -13.524, -14.445, -14.265, -15.267, -14.712, -15.531, -14.879, -16.825, 
#        -14.376, -15.595, -15.075, -14.443, -15.161, -14.428, -14.553, -14.77, 
#        -16.082, -15.321, -14.444, -16.362, -15.359, -15.168, -15.817, -14.09, 
#        -13.84, -14.297, -14.551 ][:all_nReps[isim]]
    vel = [rd.gauss(all_v_means[isim], v_std) for i in range(all_nReps[isim])]
    coeff = [[complex(1, 0), complex(0, 0)] for i in range(all_nReps[isim])]
    
    # All units must be atomic units
    ctmqc_env = {
            'pos': pos,  # Intial Nucl. pos | nrep |in bohr
            'vel': vel,  # Initial Nucl. veloc | nrep |au_v
            'C': coeff,  # Intial WF |nrep, 2| -
            'mass': [2000],  # nuclear mass |nrep| au_m
            'tullyModel': all_models[isim],  # Which model | | -
            'max_time': all_max_times[isim],  # How many steps | | -
            'dx': 1e-5,  # The increment for the NACV and grad E calc | | bohr
            'dt': dT,  # The timestep | |au_t
            'elec_steps': elecSteps,  # Num elec. timesteps per nucl. one | | -
            'do_QM_F': doCTMQC_F,
            'do_QM_C': doCTMQC_C,
            'sigma': np.ones(all_nReps[isim]) * 0.3,
                }
    
    runData = main(ctmqc_env, savePath)




# Test the adiabatic momentum
if runData.ctmqc_env['do_QM_F'] or runData.ctmqc_env['do_QM_C']:
    adMom = runData.allAdMom
    adFrc = runData.allAdFrc
    adFrcDiff = np.gradient(adMom, runData.ctmqc_env['dt'], axis=0)
    if np.mean(np.abs(adFrcDiff - adFrc)) > 1e-5:
        print("The time derivative of the adiabatic momentum doesn't give " +
              "the adiabatic force.")
        print("\n")
        print("Plotting the adiabtic force calculated in the code and the " +
              "adiabatic force calculated from the time-derivative of the " +
              "momentum. The yellow and black lines should be on top of each" +
              "other.")
        plt.plot(runData.allt, adFrc[:, :, 0], 'k')
        plt.plot(runData.allt, adFrc[:, :, 1], 'k')
        plt.plot(runData.allt, adFrcDiff[:, :, 0], 'y')
        plt.plot(runData.allt, adFrcDiff[:, :, 1], 'y')
        raise SystemExit("BREAK")




if runData.ctmqc_env['iter'] > 30:
    f, axes = plt.subplots(2)
    
    #R = runData.allR[:, 0]
    axes[0].plot(runData.allt, runData.allAdPop[:, :, 0], 'r', lw=0.2, alpha=0.3)
    axes[0].plot(runData.allt, runData.allAdPop[:, :, 1], 'b', lw=0.2, alpha=0.3)
    axes[0].plot(runData.allt, np.mean(runData.allAdPop[:, :, 0], axis=1), 'r',
                 label="Ground")
    axes[0].plot(runData.allt, np.mean(runData.allAdPop[:, :, 1], axis=1), 'b',
                 label="Excited")
    axes[0].set_ylabel("Populations (ad)")
    axes[0].legend()
    
    deco = runData.allAdPop[:, :, 0] * runData.allAdPop[:, :, 1]
    axes[1].plot(runData.allt, deco, lw=0.2, alpha=0.3)
    axes[1].plot(runData.allt, np.mean(deco, axis=1))
    axes[1].set_ylabel("Coherence")
    axes[1].set_xlabel("Time [au]")
    
    f2, axes2 = plt.subplots(2)
    norm = np.sum(runData.allAdPop, axis=2)
    axes2[0].plot(runData.allt, np.mean(norm, axis=1))
    axes2[0].set_ylabel("Norm")
    
    kinE = 0.5 * runData.ctmqc_env['mass'][0] * (runData.allv**2)
    potE = np.sum(runData.allAdPop * runData.allE, axis=2)
    axes2[1].plot(runData.allt, np.mean(kinE + potE, axis=1), 'k',
                  label="Total")
    axes2[1].plot(runData.allt, np.mean(kinE, axis=1), 'r', label="Kinetic")
    axes2[1].plot(runData.allt, np.mean(potE, axis=1), 'g', label="Potential")
    axes2[1].set_ylabel("Energy [au]")
    axes2[1].set_xlabel("Time [au]")
    axes2[1].legend()
    plt.show()
    
    #plot.plot_di_pops(runData.allt, runData.allu, "Time")
    #plot.plot_Rabi(runData.allt, runData.allH[0, 0], ctmqc_env)
    #plot.plot_ad_pops(R, runData.allAdPop)
    #plot.plot_H(runData.allt, runData.allH, "Time")
    
    #plot.plot_H_all_x(ctmqc_env)
    #plot.plot_eh_frc_all_x(ctmqc_env)
    #plot.plot_adFrc_all_x(ctmqc_env)
    #plot.plot_ener_all_x(ctmqc_env)
    #plot.plot_NACV_all_x(ctmqc_env)

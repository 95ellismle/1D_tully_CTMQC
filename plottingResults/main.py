#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:25:28 2019

@author: mellis
"""

import matplotlib.pyplot as plt
import numpy as np
import os

import getData
import plotData

norm_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/Ehrenfest_Data/NormCons_vs_ElecDT"
norm_root_ctmqc_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/CTMQC_Data/NoDC/NormCons_vs_ElecDT"
ener_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/Ehrenfest_Data/EnerCons_vs_NuclDT"
ener_root_ctmqc_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/CTMQC_Data/NoDC/EnerCons_vs_NuclDT"
pops_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/Ehrenfest_Data/Pops_Compare2"
pops_ctmqc_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/CTMQC_Data/No_DC"
Rlk_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/CTMQC_Data/No_DC"


'''Plot the Ehrenfest norm conservation for model 1, 2, 3 and 4 for the 
high momentum cases vs timestep, using the gradH NACV and a constant 
Nuclear timestep of 0.1 au.

If different parameters are required please re-run the code and point the
variable norm_root_folder to the data folder.'''
plot_norms = False

'''Plot the Ehrenfest norm conservation for model 1, 2, 3 and 4 for the 
high momentum cases vs timestep, using the gradH NACV and a constant 
Electronic timestep of 0.01 au.

If different parameters are required please re-run the code and point the
variable ener_root_folder to the data folder.'''
plot_ener_cons = False

"""
Plot the high and low momentum cases of model 1, 2, 3 and 4 and compare them to
the Gossel and Agostini (Ehrenfest) data. 
If the Agostini and Gossel data have different initial momenta then it will
choose Gossel.
"""
plot_pop_lit_compare = False

"""
Will plot a graph of the populations, the norm, and the Rlk on the same figure
with the axes lined up in time. This is useful in checking whether the Rlk
spikes are causing problems in the populations or forces.
"""
plot_pop_norm_Rlk = False

"""
Will plot a graph of the forces, the ener, and the Rlk on the same figure
with the axes lined up in time. This is useful in checking whether the Rlk
spikes are causing problems in the populations or forces.
"""
plot_frc_ener_Rlk = False

"""
Plot the high and low momentum cases of model 1, 2, 3 and 4 and compare them to
the Gossel and Agostini (CTMQC) data. 
If the Agostini and Gossel data have different initial momenta then it will
choose Gossel.
"""
plot_pop_lit_compare_ctmqc = False

'''Plot the Ehrenfest norm conservation for model 1, 2, 3 and 4 for the 
high momentum cases vs timestep, using the gradPhi NACV and a constant 
Nuclear timestep of 0.1 au.

If different parameters are required please re-run the code and point the
variable norm_root_folder to the data folder.'''
plot_norms_ctmqc = True

###############################################################################

if plot_norms:
    allData = getData.NestedSimData(norm_root_folder, ['|C|^2', 'time'])
    
    fa = plt.subplots(2, 2)
    
    model = 1
    for x in range(2):
        for y in range(2):
            ax = fa[1][x][y]
            ax.set_title("Model %i" % model, fontsize=25)
            plotData.plotNormVsElecDt(allData, model, fa[0], ax)
            if x == 0:
                ax.set_xlabel("")
            if y == 1:
                ax.set_ylabel("")
            model += 1
            
    plt.tight_layout()
    plt.show()


###############################################################################


if plot_norms_ctmqc:
    allData = getData.NestedSimData(norm_root_ctmqc_folder, ['|C|^2', 'time'])
    
    fa = plt.subplots(2, 2)
    
    model = 1
    for x in range(2):
        for y in range(2):
            ax = fa[1][x][y]
            ax.set_title("Model %i" % model, fontsize=25)
            plotData.plotNormVsElecDt(allData, model, fa[0], ax)
            if x == 0:
                ax.set_xlabel("")
            if y == 1:
                ax.set_ylabel("")
            model += 1
            
    plt.tight_layout()
    plt.show()


###############################################################################


if plot_ener_cons:
    allData = getData.NestedSimData(ener_root_folder,
                                    ["time", 'ener', "|C|^2", "vel"])
    
    fa = plt.subplots(2, 2)
    
    model = 1
    for x in range(2):
        for y in range(2):
            ax = fa[1][x][y]
            ax.set_title("Model %i" % model, fontsize=25)
            plotData.plotEnerVsNuclDt(allData, model, fa[0], ax)
            if x == 0:
                ax.set_xlabel("")
            if y == 1:
                ax.set_ylabel("")
            model += 1
    
    plt.tight_layout()
    plt.show()
   
    
###############################################################################
    
    
if plot_pop_lit_compare:
    allData = getData.NestedSimData(pops_root_folder, ['time', '|C|^2'])
    fredData = getData.FredericaData()
    gossData = getData.GosselData()
    
    for mom in ['high', 'low']:
        model = 1
        f, a = plt.subplots(2, 2)
        fD, aD = plt.subplots(2, 2)
    
    
        for x in range(2):
            for y in range(2):
                ax = a[x][y]
                axD = aD[x][y]
                dfFredDeco = getattr(fredData,
                                     "mod%i_%sMom_deco" % (model, mom))
                dfFredPop = getattr(fredData,
                                    "mod%i_%sMom_pops" % (model, mom))
                fredMom = getattr(fredData,
                                  "mod%i_%sMom" % (model, mom))
    
                dfGossDeco = getattr(gossData,
                                     "mod%i_%sMom_deco" % (model, mom))
                dfGossPop = getattr(gossData,
                                    "mod%i_%sMom_pops" % (model, mom))
                gossMom = getattr(gossData,
                                  "mod%i_%sMom" % (model, mom))
    
                myData = allData.query_data({'tullyModel': model,
                                             'velInit': gossMom * 5e-4})

                myPops = [d.adPop for d in myData]
                if len(myPops) > 0:
                    if sum(np.diff([len(pops) for pops in myPops])) == 0:
                        myCoh = [pops[:, :, 0]  * pops[:, :, 1]
                                 for pops in myPops]
                        myCoh = np.mean(myCoh, axis=2)
                        myCoh = np.mean(myCoh, axis=0)
                        myPops = [np.mean(pops, axis=1) for pops in myPops]
                        myPops = np.mean(myPops, axis=0)
                    else:
                        myPops = []
    
                # Plot the populations
                ax.plot(dfGossPop['Eh_x']*0.024188843265857,
                        dfGossPop['Eh_y'], label="Gossel, 18", color='k')
                if len(myPops) > 0:
                    ax.plot(myData[0].times*0.024188843265857, myPops[:, 0],
                            label="My Data", color='r')
                if gossMom == fredMom:
                    ax.plot(dfFredPop['Eh_x']*0.024188843265857,
                            dfFredPop['Eh_y'], label="Agostini, 16", color='g')
    
                xlimmax = max(np.nanmax(dfGossPop['Eh_x']),
                              np.nanmax(dfFredPop['Eh_x'])
                              )*0.024188843265857*1.1
                ax.set_xlim([-1.9, xlimmax])
                
                if x == 1:
                    ax.set_xlabel("Timestep [fs]")
                if y == 0:
                    ax.set_ylabel("Ad. Pop.")
                ax.set_title("Model %i" % model, fontsize=24)
    
                ax.annotate(r"P$_{0}$: %.2g au" % gossMom, (0.04, 0.06),
                            xycoords='axes fraction', fontsize=15)
                ax.legend()

                # Plot the coherences
                axD.plot(dfGossDeco['Eh_x']*0.024188843265857,
                        dfGossDeco['Eh_y'], label="Gossel, 18", color='k')
                if len(myPops) > 0:
                    axD.plot(myData[0].times*0.024188843265857, myCoh,
                            label="My Data", color='r')
                if gossMom == fredMom:
                    axD.plot(dfFredDeco['Eh_x']*0.024188843265857,
                            dfFredDeco['Eh_y'], label="Agostini, 16", color='g')
    
                xlimmax = max(np.nanmax(dfGossDeco['Eh_x']),
                              np.nanmax(dfFredDeco['Eh_x'])
                              )*0.024188843265857*1.1
                axD.set_xlim([-1.9, xlimmax])
                
                if x == 1:
                    axD.set_xlabel("Timestep [fs]")
                if y == 0:
                    axD.set_ylabel("Coherence")
                axD.set_title("Model %i" % model, fontsize=24)
    
                axD.annotate(r"P$_{0}$: %.2g au" % gossMom, (0.04, 0.06),
                            xycoords='axes fraction', fontsize=15)
                axD.legend()
                model += 1
    
        f.suptitle("%s Momentum -Populations" % mom.title(), fontsize=30)
        plt.tight_layout()
        
        fD.suptitle("%s Momentum -Coherences" % mom.title(), fontsize=30)
        plt.tight_layout()
        
        
        
###############################################################################       
        
        
        
if plot_pop_norm_Rlk:
    allData = getData.NestedSimData(Rlk_root_folder,
                                     ['time', '|C|^2', "Rlk", "RI0"])
    for model in range(1, 2):
        for mom in ['low']:#, 'high']:
            modelData = allData.query_data({'tullyModel': model})
            initVels = [data.velInit for data in modelData]
            if mom == 'high':
                modelData = [d for d in modelData if d.velInit == max(initVels)]
            elif mom == 'low':
                modelData = [d for d in modelData if d.velInit == min(initVels)]
            
            
            for iSim, data in enumerate(modelData):
                data.times *= 0.024188843265857
                f, a = plt.subplots(3)
                
                alpha, lw = 0.1, 0.2
                
                # Plot Norms
                norm = np.sum(data.adPop, axis=2)
                a[0].plot(data.times, norm, 'r-', alpha=alpha, lw=lw)
                a[0].plot(data.times, np.mean(norm, axis=1), 'r-')
                a[0].set_ylabel("Norm")
                
                # Plot Pops
                a[1].plot(data.times, data.adPop[:, :, 0], 'r-',
                          alpha=alpha, lw=lw)
                a[1].plot(data.times, np.mean(data.adPop[:, :, 0], axis=1),
                          'r-')
                a[1].plot(data.times, data.adPop[:, :, 1], 'b-',
                          alpha=alpha, lw=lw)
                a[1].plot(data.times, np.mean(data.adPop[:, :, 1], axis=1),
                          'b-')
                a[1].set_ylabel("Ad. Pop.")
                
                # Plot Rlk
                a[2].plot(data.times, data.Rlk[:, 0, 1], 'k-')
                a[2].plot(data.times, data.RI0, 'k--', lw=lw, alpha=alpha)
                a[2].set_ylabel("Intercept [bohr]")
                
                savePath = "/homes/mellis/Documents/Graphs/Tully_Models/" + \
                            "CTMQC_Final_Data/Rlk_Story/AllNorm_Pop_Rlk"
                savePath = savePath + "/Model_%i/%sMom/" % (model, mom)
                if not os.path.isdir(savePath): os.makedirs(savePath)
                
                savePath = savePath + "/Repeat_%i" % iSim
#                f.savefig(savePath)
#                plt.close("all")
        
        
        
###############################################################################       
        
        
        
if plot_frc_ener_Rlk:
    allData = getData.NestedSimData(Rlk_root_folder,
                                     ['time', '|C|^2', 'E', 'v', "Rlk",
                                      'tot force'])
    for model in range(1, 5):
        for mom in ['low', 'high']:
            modelData = allData.query_data({'tullyModel': model})
            initVels = [data.velInit for data in modelData]
            if mom == 'high':
                modelData = [d for d in modelData if d.velInit == max(initVels)]
            elif mom == 'low':
                modelData = [d for d in modelData if d.velInit == min(initVels)]
    
            for iSim, data in enumerate(modelData):
                data.times *= 0.024188843265857
                f, a = plt.subplots(3)
                
                alpha, lw = 0.1, 0.2
                
                # Plot Total Energy
                Epot = np.sum(data.adPop * data.E, axis=2)
                Ekin = 1000 * (data.v**2)
                Etot = Epot + Ekin
                a[0].plot(data.times, Etot, 'k-', alpha=alpha, lw=lw)
                a[0].plot(data.times, np.mean(Etot, axis=1), 'k-')
                a[0].set_ylabel("Tot. E [Ha]")
                
                # Plot Forces
                a[1].plot(data.times, data.F, 'k-', alpha=alpha, lw=lw)
                a[1].plot(data.times, np.mean(data.F, axis=1), 'k-')
                a[1].set_ylabel(r"F$_{tot}$ [$\frac{Ha}{Bohr}$]")
                
                # Plot Rlk
                a[2].plot(data.times, data.Rlk[:, 0, 1], 'k-')
                a[2].set_ylabel("Rlk [bohr]")
                
                savePath = "/homes/mellis/Documents/Graphs/Tully_Models/" + \
                            "CTMQC_Final_Data/Rlk_Story/AllEner_Frc_Rlk"
                savePath = savePath + "/Model_%i/%sMom/" % (model, mom)
                if not os.path.isdir(savePath): os.makedirs(savePath)
                
                savePath = savePath + "/Repeat_%i" % iSim
                f.savefig(savePath)
                plt.close("all")
    
    
###############################################################################
    
    
if plot_pop_lit_compare_ctmqc:
    allData = getData.NestedSimData(pops_ctmqc_root_folder, ['time', '|C|^2'])
    fredData = getData.FredericaData()
    gossData = getData.GosselData()
    
    for mom in ['high', 'low']:
        model = 1
        f, a = plt.subplots(2, 2)
        fD, aD = plt.subplots(2, 2)
    
    
        for x in range(2):
            for y in range(2):
                ax = a[x][y]
                axD = aD[x][y]
                dfFredDeco = getattr(fredData,
                                     "mod%i_%sMom_deco" % (model, mom))
                dfFredPop = getattr(fredData,
                                    "mod%i_%sMom_pops" % (model, mom))
                fredMom = getattr(fredData,
                                  "mod%i_%sMom" % (model, mom))
    
                dfGossDeco = getattr(gossData,
                                     "mod%i_%sMom_deco" % (model, mom))
                dfGossPop = getattr(gossData,
                                    "mod%i_%sMom_pops" % (model, mom))
                gossMom = getattr(gossData,
                                  "mod%i_%sMom" % (model, mom))
    
                myData = allData.query_data({'tullyModel': model,
                                             'velInit': gossMom * 5e-4})

                myPops = [d.adPop for d in myData]
                if len(myPops) > 0:
                    all_lens = [len(pops) for pops in myPops]
                    pop_mask = [i for i, pops in enumerate(myPops)
                                if len(pops) == max(all_lens)]
                    myPops = [myPops[i] for i in pop_mask]
                    
                    myCoh = [pops[:, :, 0]  * pops[:, :, 1]
                             for pops in myPops]
                    myCoh = np.mean(myCoh, axis=2)
                    myCoh = np.mean(myCoh, axis=0)
                    myPops = [np.mean(pops, axis=1) for pops in myPops]
                    myPops = np.mean(myPops, axis=0)

                # Plot the populations
                ax.plot(dfGossPop['CTMQC_x']*0.024188843265857,
                        dfGossPop['CTMQC_y'], label="Gossel, 18", color='k')
                if len(myPops) > 0:
                    ax.plot(myData[pop_mask[0]].times*0.024188843265857,
                            myPops[:, 0], label="My Data", color='r')
                for data in allData.query_data({'tullyModel': model,
                                             'velInit': gossMom * 5e-4}):
                    ax.plot(data.times*0.024188843265857,
                            np.mean(data.adPop, axis=1)[:, 0], 'r', lw=0.1)
                if gossMom == fredMom:
                    ax.plot(dfFredPop['CTMQC_x']*0.024188843265857,
                            dfFredPop['CTMQC_y'], label="Agostini, 16", color='g')
    
                xlimmax = max(np.nanmax(dfGossPop['CTMQC_x']),
                              np.nanmax(dfFredPop['CTMQC_x'])
                              )*0.024188843265857*1.1
                ax.set_xlim([-1.9, xlimmax])
                
                if x == 1:
                    ax.set_xlabel("Timestep [fs]")
                if y == 0:
                    ax.set_ylabel("Ad. Pop.")
                ax.set_title("Model %i" % model, fontsize=24)
    
                ax.annotate(r"P$_{0}$: %.2g au" % gossMom, (0.04, 0.06),
                            xycoords='axes fraction', fontsize=15)
                ax.legend(fontsize=18)

                # Plot the coherences
                axD.plot(dfGossDeco['CTMQC_x']*0.024188843265857,
                        dfGossDeco['CTMQC_y'], label="Gossel, 18", color='k')
                if len(myPops) > 0:
                    axD.plot(myData[pop_mask[0]].times*0.024188843265857, myCoh,
                            label="My Data", color='r')
                for data in allData.query_data({'tullyModel': model,
                                             'velInit': gossMom * 5e-4}):
                    axD.plot(data.times*0.024188843265857,
                            np.mean(data.adPop[:, :, 0] * data.adPop[:, :, 1],
                                    axis=1),
                            'r', lw=0.1)
                if gossMom == fredMom:
                    axD.plot(dfFredDeco['CTMQC_x']*0.024188843265857,
                            dfFredDeco['CTMQC_y'], label="Agostini, 16", color='g')
    
                xlimmax = max(np.nanmax(dfGossDeco['CTMQC_x']),
                              np.nanmax(dfFredDeco['CTMQC_x'])
                              )*0.024188843265857*1.1
                axD.set_xlim([-1.9, xlimmax])
                
                if x == 1:
                    axD.set_xlabel("Timestep [fs]")
                if y == 0:
                    axD.set_ylabel("Coherence")
                axD.set_title("Model %i" % model, fontsize=24)
    
                axD.annotate(r"P$_{0}$: %.2g au" % gossMom, (0.04, 0.06),
                            xycoords='axes fraction', fontsize=15)
                axD.legend(fontsize=18)
                model += 1

    
        f.suptitle("%s Momentum -Populations" % mom.title(), fontsize=30)
        f.subplots_adjust(top=0.899,
                            bottom=0.099,
                            left=0.078,
                            right=0.989,
                            hspace=0.269,
                            wspace=0.152)
        
        fD.suptitle("%s Momentum -Coherences" % mom.title(), fontsize=30)
        fD.subplots_adjust(top=0.899,
                            bottom=0.099,
                            left=0.078,
                            right=0.989,
                            hspace=0.269,
                            wspace=0.152)

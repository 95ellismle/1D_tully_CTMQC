from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:25:28 2019

@author: mellis
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import re

import getData
import plotData

norm_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/Ehrenfest_Data/NormCons_vs_ElecDT"
norm_root_ctmqc_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/CTMQC_Data/With_Ehren_DC/NormCons_vs_ElecDT"
ener_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/Ehrenfest_Data/EnerCons_vs_NuclDT"
ener_root_ctmqc_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/CTMQC_Data/NoDC/EnerCons_vs_NuclDT"
pops_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/Ehrenfest_Data/Pops_Compare2"
pops_ctmqc_root_folder = "/homes/mellis/Documents/Code_bits_and_bobs/1D_tully_model/test/Sig_0.2/Kinit_30/Repeat_19"
pops_ctmqc_DC_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/CTMQC_Data/With_Extrap_DC/Pops_Compare"
Rlk_root_folder = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/CTMQC_Data/With_DC/Pops_Compare"


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
plot_pop_lit_compare_ctmqc = True

'''Plot the CTMQC norm conservation for model 1, 2, 3 and 4 for the 
high momentum cases vs timestep, using the gradPhi NACV and a constant 
Nuclear timestep of 0.1 au.

If different parameters are required please re-run the code and point the
variable norm_root_folder to the data folder.'''
plot_norms_ctmqc = False

'''Plot the CTMQC norm conservation for model 1, 2, 3 and 4 for the 
high momentum cases vs timestep, using the gradPhi NACV and a constant 
Nuclear timestep of 0.1 au.

If different parameters are required please re-run the code and point the
variable norm_root_folder to the data folder.'''
plot_ener_cons_ctmqc = False

# Whether or not to compare the Non-divergence corrected and diveregence
#  corrected data.
compare_DC = False
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
    lab1 = "No DC Corr"
    if compare_DC:
        if len(re.findall("With_.*_DC", norm_root_ctmqc_folder)) > 0:
            folderPath = re.sub("With.*_DC", "NoDC", norm_root_ctmqc_folder)
            lab, col, lab1, col1 = 'No DC Corr', 'r', "Ehren DC", 'b'
        else:
            folderPath = norm_root_ctmqc_folder.replace("NoDC", "With_DC")
            lab, col, lab1, col1 = 'With DC Corr', 'b', "No DC Corr", 'r'
        allData2 = getData.NestedSimData(folderPath, ['|C|^2', 'time'])
#        allData3 = getData.NestedSimData(folderPath.replace("NoDC", "With_DC"),
#                                         ['|C|^2', 'time'])
#        allData4 = getData.NestedSimData(folderPath.replace("NoDC", "With_Extrap_DC"),
#                                         ['|C|^2', 'time'])

    fa = plt.subplots(2, 2)
    
    color = 'k'
    model = 1
    for x in range(2):
        for y in range(2):
            ax = fa[1][x][y]
            ax.set_title("Model %i" % model, fontsize=25)

#            plotData.plotNormVsElecDt(allData3, model, fa[0], ax,
#                                          {'color': 'g'}, 'RI0 DC')
#            plotData.plotNormVsElecDt(allData4, model, fa[0], ax,
#                                          {'color': 'k'}, 'Extrap DC')
            if compare_DC:
                plotData.plotNormVsElecDt(allData2, model, fa[0], ax,
                                          {'color': col}, lab)
                color = col1
            plotData.plotNormVsElecDt(allData, model, fa[0], ax,
                                      {'color': color}, lab1)
            if compare_DC:
                ax.legend()
            if x == 0:
                ax.set_xlabel("")
            if y == 1:
                ax.set_ylabel("")
            model += 1
            
    plt.tight_layout()
    fa[0].savefig('/homes/mellis/tmp.png')


###############################################################################


if plot_ener_cons:
    allData = getData.NestedSimData(ener_root_folder,
                                    ["time", 'ener', "|C|^2", "vel"])
    
    fa = plt.subplots(2, 2)
    
    model = 1
    color='k'
    for x in range(2):
        for y in range(2):
            ax = fa[1][x][y]
            ax.set_title("Model %i" % model, fontsize=25)
            if x == 0:
                ax.set_xlabel("")
            if y == 1:
                ax.set_ylabel("")
            model += 1
    
    plt.tight_layout()
    plt.show()
   
    
###############################################################################


if plot_ener_cons_ctmqc:
    allData = getData.NestedSimData(ener_root_ctmqc_folder,
                                    ["time", 'ener', "|C|^2", "vel"])
    if compare_DC:
        if 'With_DC' in ener_root_ctmqc_folder:
            folderPath = ener_root_ctmqc_folder.replace("With_DC", "NoDC")
        else:
            folderPath = ener_root_ctmqc_folder.replace("NoDC", "With_DC")
        
        allData2 = getData.NestedSimData(folderPath, ["time", 'ener', "|C|^2", "vel"])
    
    fa = plt.subplots(2, 2)
    
    model = 1
    for x in range(2):
        for y in range(2):
            ax = fa[1][x][y]
            ax.set_title("Model %i" % model, fontsize=25)
            if compare_DC:
                plotData.plotEnerVsNuclDt(allData2, model, fa[0], ax,
                                          {'color': 'b'}, 'With DC Corr')
                color='r'

            plotData.plotEnerVsNuclDt(allData, model, fa[0], ax,
                                      {'color': color}, 'No DC Corr')
            if x == 0:
                ax.set_xlabel("")
            if y == 1:
                ax.set_ylabel("")
            if compare_DC:
                ax.legend()
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
    with_DC = 'With_DC' in Rlk_root_folder
    allData = getData.NestedSimData(Rlk_root_folder,
                                     ['time', '|C|^2', "Rlk", "RI0", "effR"])
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
                
                # Plot Norms
#                if not with_DC:
                norm = np.sum(data.adPop, axis=2)
                a[0].plot(data.times, norm, 'r-', alpha=alpha, lw=lw)
                a[0].plot(data.times, np.mean(norm, axis=1), 'r-')
                a[0].set_ylabel("Norm")
                
                popAx = 1
                # Plot Pops
                a[popAx].plot(data.times, data.adPop[:, :, 0], 'r-',
                          alpha=alpha, lw=lw)
                a[popAx].plot(data.times, np.mean(data.adPop[:, :, 0], axis=1),
                          'r-')
                a[popAx].plot(data.times, data.adPop[:, :, 1], 'b-',
                          alpha=alpha, lw=lw)
                a[popAx].plot(data.times, np.mean(data.adPop[:, :, 1], axis=1),
                          'b-')
                a[popAx].set_ylabel("Ad. Pop.")
                
                # Plot Rlk
                RlkAx = 2
                a[RlkAx].plot(data.times, data.Rlk[:, 0, 1], 'k-',
                          label=r"$\mathbf{R}_{lk, \nu}$")
                a[RlkAx].plot(data.times, data.RI0[:, 0], 'k--', alpha=alpha, lw=lw)
                a[RlkAx].plot(data.times, data.RI0[:, 1:], 'k--', lw=lw, alpha=alpha)
                a[RlkAx].plot(data.times, data.effR[:, 0, 1], 'g-')
                a[RlkAx].set_ylabel("Rlk [bohr]")
                a[RlkAx].set_xlabel("Timestep [fs]")
                
                RlkArtist = plt.Line2D((0,1),(0,0), color='k')
                effRArtist = plt.Line2D((0,1),(0,0), color='g')
                RI0Artist = plt.Line2D((0,1),(0,0), color='k', linestyle='--',
                                       lw=0.7, alpha=0.7)
                artists = [RlkArtist, RI0Artist, effRArtist]
                labels = [r"$\mathbf{R}_{lk, \nu}$",
                          r"$\mathbf{R}_{0, \nu}^{(I)}$", r"$effR_{\nu}$"]
                a[RlkAx].legend(artists, labels)
                a[RlkAx].set_ylim([np.min(data.RI0)*1.1,
                                   np.max(data.RI0)*1.1])
                
                if with_DC:
                    savePath = "/homes/mellis/Documents/Graphs/Tully_Models/" + \
                                "CTMQC_Final_Data/DC_Corr/AllNorm_Pop_Rlk"
                else:
                    savePath = "/homes/mellis/Documents/Graphs/Tully_Models/" + \
                                "CTMQC_Final_Data/Bad_Rlk/AllNorm_Pop_Rlk"
                savePath = savePath + "/Model_%i/%sMom/" % (model, mom)
                if not os.path.isdir(savePath): os.makedirs(savePath)
                
                savePath = savePath + "/Repeat_%i" % iSim
                f.savefig(savePath)
                plt.close("all")
                
                print("\rRepeat %i done" % iSim, end="\r")
            print("Model %i %s momentum done" % (model, mom))
        
        
        
###############################################################################       
        
        
        
if plot_frc_ener_Rlk:
    with_DC = 'With_DC' in Rlk_root_folder
    allData = getData.NestedSimData(Rlk_root_folder,
                                     ['time', '|C|^2', 'E', 'v', "Rlk",
                                      'tot force', "RI0", "effR"])
    for model in range(1, 5):
        for mom in ['high', 'low']:
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
                frcAx = 1
                a[frcAx].plot(data.times, data.F, 'k-', alpha=alpha, lw=lw)
                a[frcAx].plot(data.times, np.mean(data.F, axis=1), 'k-')
                a[frcAx].set_ylabel(r"F$_{tot}$ [$\frac{Ha}{Bohr}$]")
                
                # Plot Rlk
                RlkAx = 2
                a[RlkAx].plot(data.times, data.Rlk[:, 0, 1], 'k-',
                          label=r"$\mathbf{R}_{lk, \nu}$")
                a[RlkAx].plot(data.times, data.RI0[:, 0], 'k--', alpha=alpha, lw=lw)
                a[RlkAx].plot(data.times, data.RI0[:, 1:], 'k--', lw=lw, alpha=alpha)
                a[RlkAx].plot(data.times, data.effR[:, 0, 1], 'g-')
                a[RlkAx].set_ylabel("Rlk [bohr]")
                a[RlkAx].set_xlabel("Timestep [fs]")
                
                RlkArtist = plt.Line2D((0,1),(0,0), color='k')
                effRArtist = plt.Line2D((0,1),(0,0), color='g')
                RI0Artist = plt.Line2D((0,1),(0,0), color='k', linestyle='--',
                                       lw=0.7, alpha=0.7)
                artists = [RlkArtist, RI0Artist, effRArtist]
                labels = [r"$\mathbf{R}_{lk, \nu}$",
                          r"$\mathbf{R}_{0, \nu}^{(I)}$", r"$effR_{\nu}$"]
                a[RlkAx].legend(artists, labels)
                a[RlkAx].set_ylim([np.min(data.RI0)*1.1,
                                   np.max(data.RI0)*1.1])
                
                if with_DC:
                    savePath = "/homes/mellis/Documents/Graphs/Tully_Models/" + \
                                "CTMQC_Final_Data/DC_Corr/AllEner_Frc"
                else:
                    savePath = "/homes/mellis/Documents/Graphs/Tully_Models/" + \
                                "CTMQC_Final_Data/Bad_Rlk/AllEner_Frc_Rlk"
                savePath = savePath + "/Model_%i/%sMom/" % (model, mom)
                if not os.path.isdir(savePath): os.makedirs(savePath)
                
                savePath = savePath + "/Repeat_%i" % iSim
                f.savefig(savePath)
                plt.close("all")
                            
                print("\rRepeat %i done" % iSim, end="\r")
            print("Model %i %s momentum done" % (model, mom))

    
###############################################################################
    
    
if plot_pop_lit_compare_ctmqc:
    
    def get_myPops_myCoh(myData):
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
            return myPops, myCoh, pop_mask
        return False, False, False
    
    def plot_myData(myData, legend="My Data", color='#ff0000'):
        myPops, myCoh, pop_mask = get_myPops_myCoh(myData)
        
        darkenColor = 0.7
        colCodes = [color.strip('#')[i*2:i*2+2] for i in range(3)]
        color1 = [hex(int(eval('0x%s'%i) * darkenColor)).strip('0x')
                  for i in colCodes]
        color1 = [i if len(i) > 1 else '00' for i in color1]
        color1 = '#'+''.join(color1)
        # Plot the populations
        if myPops is not False:
            ax.plot(myData[pop_mask[0]].times*0.024188843265857,
                    myPops[:, 0], label=legend, color=color)
            for data in myData:
                ax.plot(data.times*0.024188843265857,
                        np.mean(data.adPop, axis=1)[:, 0], color1, lw=0.6,
                        alpha=0.4)
        # Plot the Coherences
        if myCoh is not False:
            axD.plot(myData[pop_mask[0]].times*0.024188843265857,
                    myCoh, label=legend, color=color)
            for data in myData:
                coherences = data.adPop[:, :, 0] * data.adPop[:, :, 1] 
                axD.plot(data.times*0.024188843265857,
                        np.mean(coherences, axis=1), color1, lw=0.6,
                        alpha=0.4)

    def get_ExtData(extData, model, mom):
        dfDeco = getattr(extData,
                             "mod%i_%sMom_deco" % (model, mom))
        dfPop = getattr(extData,
                            "mod%i_%sMom_pops" % (model, mom))
        mom = getattr(extData,
                          "mod%i_%sMom" % (model, mom))
        return dfPop, dfDeco, mom

    
    allData = getData.NestedSimData(pops_ctmqc_root_folder, ['time', '|C|^2'])
    if compare_DC:
        allDCDataEx = getData.NestedSimData(pops_ctmqc_DC_root_folder,
                                          ['time', '|C|^2'])
        allDCDataEh = getData.NestedSimData(pops_ctmqc_DC_root_folder.replace("Extrap", "Ehren"),
                                          ['time', '|C|^2'])
        allDCDataRI0 = getData.NestedSimData(pops_ctmqc_DC_root_folder.replace("Extrap_", ""),
                                          ['time', '|C|^2'])
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

                dfFredPop, dfFredDeco, fredMom = get_ExtData(fredData, model,
                                                             mom)
                dfGossPop, dfGossDeco, gossMom = get_ExtData(gossData, model,
                                                             mom)

    
                myData = allData.query_data({'tullyModel': model,
                                             'velInit': gossMom * 5e-4})
                plot_myData(myData)
                if compare_DC:
                    for data, col, lab in zip((allDCDataEx, allDCDataEh, allDCDataRI0),
                                         ('#0000ff', '#555555', '#00bb00'),
                                         ('Extrap DC', 'Ehren DC', 'RI0 DC')):
                       myDCData = data.query_data({'tullyModel': model,
                                                     'velInit': gossMom * 5e-4})
                       plot_myData(myDCData, lab, color=col)

                
                ax.plot(dfGossPop['CTMQC_x']*0.024188843265857,
                        dfGossPop['CTMQC_y'], label="Gossel, 18", color='k')    
                ax.plot(dfGossPop['exact_x']*0.024188843265857,
                        dfGossPop['exact_y'], label="Exact", color='b')
#                if gossMom == fredMom:
#                    ax.plot(dfFredPop['CTMQC_x']*0.024188843265857,
#                            dfFredPop['CTMQC_y'], label="Agostini, 16", color='g')
    
                xlimmax = max(np.nanmax(dfGossPop['CTMQC_x']),
                              np.nanmax(dfFredPop['CTMQC_x'])
                              )*0.024188843265857*1.1
                ax.set_xlim([-1.9, xlimmax])
                
                if x == 1:
                    ax.set_xlabel("Sim. Time [fs]")
                if y == 0:
                    ax.set_ylabel("Ad. Pop.")
                ax.set_title("Model %i" % model, fontsize=24)
    
                ax.annotate(r"P$_{0}$: %.2g au" % gossMom, (0.04, 0.06),
                            xycoords='axes fraction', fontsize=15)
                ax.legend(fontsize=18)

                # Plot the coherences
                axD.plot(dfGossDeco['CTMQC_x']*0.024188843265857,
                        dfGossDeco['CTMQC_y'], label="Gossel, 18", color='k')
                axD.plot(dfGossDeco['exact_x']*0.024188843265857,
                        dfGossDeco['exact_y'], label="Exact", color='b')
#                if gossMom == fredMom:
#                    axD.plot(dfFredDeco['CTMQC_x']*0.024188843265857,
#                            dfFredDeco['CTMQC_y'], label="Agostini, 16", color='g')
    
                xlimmax = max(np.nanmax(dfGossDeco['CTMQC_x']),
                              np.nanmax(dfFredDeco['CTMQC_x'])
                              )*0.024188843265857*1.1
                axD.set_xlim([-1.9, xlimmax])
                
                if x == 1:
                    axD.set_xlabel("Sim. Time [fs]")
                if y == 0:
                    axD.set_ylabel("Coherence")
                axD.set_title("Model %i" % model, fontsize=24)
    
                axD.annotate(r"P$_{0}$: %.2g au" % gossMom, (0.04, 0.06),
                            xycoords='axes fraction', fontsize=15)
                axD.legend(fontsize=18)
                model += 1

    
        f.suptitle("%s Momentum -Populations" % mom.title(), fontsize=30)
        f.subplots_adjust(top=0.884,
                            bottom=0.109,
                            left=0.073,
                            right=0.979,
                            hspace=0.269,
                            wspace=0.152)
        
        fD.suptitle("%s Momentum -Coherences" % mom.title(), fontsize=30)
        fD.subplots_adjust(top=0.884,
                            bottom=0.109,
                            left=0.073,
                            right=0.979,
                            hspace=0.269,
                            wspace=0.152)
        fD.savefig('/homes/mellis/tmpD%s.png' % mom)
        f.savefig('/homes/mellis/tmp%s.png' % mom)

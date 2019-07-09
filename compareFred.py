#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Will plot the tmpData from the python simulations (CTMQC ones -tully models)

Created on Wed Jun 12 09:38:12 2019

@author: mellis
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

mroot_folder = '/scratch/mellis/TullyModelData/FullCTMQC'
model = 1
whichPlot = 'compFred'
numStd = 1
std_or_allSim = "std"
rm_bad_sims = False


colors = {'CTMQC': (0, 0.5, 0), 'Eh': (0, 0.5, 1), 'MQC': {1, 1, 0},
          'SH': (0, 0, 1), 'exact': (1, 0, 0)}
folderNames = {'Eh': ['Ehren'], 'CTMQC': ['CTMQC']}

plotFred = ['CTMQC', 'exact']
plotMe = ['CTMQC']


frootfolder = "/homes/mellis/Documents/Graphs/Tully_Models"
# First plot Frederica's tmpData
def plotFredData(rootFolder, model, momentum=False,
                 plot_names = ['exact', 'SH', 'Eh', 'CTMQC']):
    """
    Will plot Frederica's tmpData and return the 2 (figure, axes) objects
    """
    if momentum is False:
        things =  (rootFolder, model)
        rootFolder = "%s/Model%i" % things
        allFolders = os.listdir(rootFolder)
        allFolders = [re.findall("CTMQC_\d*K", fold) for fold in allFolders]
        allFolders = ['%s/%s' % (rootFolder, i[0])
                      for i in allFolders if len(i) == 1]
        moms = [int(re.findall("\d*K", folder)[0].strip('K'))
                for folder in allFolders]
        allFolders = [f[1] for f in sorted(zip(moms, allFolders))]
    
    figure, axes = plt.subplots(2, 2)
    for iAy, folder in enumerate(allFolders):
        names = ['exact', 'SH', 'Eh', 'MQC', 'CTMQC']
        mom = re.findall("\d*K", folder)[0].strip('K')
        popsFilePath = folder + "/Pops.csv"
        decoFilePath = folder + "/Deco.csv"
        for iAx, fPath in enumerate((popsFilePath, decoFilePath)):
            with open(fPath, 'r') as f:
                lines = f.read().split('\n')[2:]
                headers = [name + j for name in names for j in ['_x', '_y']]
                txt = [','.join(headers)] + lines
            
            with open("tmp.csv", 'w') as f:
                f.write('\n'.join(txt))
                
            tmpData = pd.read_csv("tmp.csv")
        
            # Sort by x tmpData
            for name in names:
                xname, yname = name+'_x', name+'_y'
                tmpData[yname][tmpData[yname] > 1] = 1
                tmpData[yname][tmpData[yname] < 0] = 0
                
            
            for iN, name in enumerate(plot_names):
                col = colors[name]
    
                xname, yname = name+'_x', name+'_y'
                xdata, ydata = tmpData[xname], tmpData[yname]
            
                sortedData = sorted(zip(xdata, ydata))
                xdata = np.array([i[0] for i in sortedData])
                ydata = np.array([i[1] for i in sortedData])
                
                axes[iAx, iAy].plot(xdata, ydata, color=col,
                                    ls='-', lw=5, alpha=0.3)
                
                if iAx == 0 and iN == 0:
                    xPos = np.nanmax(xdata) / 12
                    axes[iAx, iAy].annotate(r"k$_0$ = %s a.u." % mom,
                                            (xPos, 0.7), fontsize=15)
                    
#                    axes[iAx, iAy].plot(xdata, 1-ydata, color=col,
#                                    ls='-', lw=5, alpha=0.3)
    
    for iy in range(2):
        for ix in range(2):
            axes[ix, iy].grid(False)
            if ix == 1:
                axes[ix, iy].set_xlabel("time (a.u)", fontsize=20)
                if iy == 0:
                    axes[ix, iy].set_ylabel("coherence", fontsize=20)
            
            else:
                axes[ix, iy].set_ylim([0, 1])
                axes[ix, iy].set_yticks([0, 0.5, 1.0])
                if iy == 0:
                    axes[ix, iy].set_ylabel("populations", fontsize=20)
            

    os.remove("tmp.csv")
        
    return figure, axes


if 'compFred' in whichPlot:
    figure, pdAxes = plotFredData(frootfolder, model,
                                  plot_names = plotFred)
    plt.suptitle("Model %i" % model, fontsize=24)
    plt.subplots_adjust(top=0.925,
                    bottom=0.1,
                    left=0.07,
                    right=0.995,
                    hspace=0.09,
                    wspace=0.185)

if 'normEner' in whichPlot:
    f, ax = plt.subplots(2, 2)
    plt.suptitle("Model %i" % model, fontsize=24)
    plt.subplots_adjust(top=0.925,
                    bottom=0.1,
                    left=0.1,
                    right=0.995,
                    hspace=0.09,
                    wspace=0.185)


def is_bad_sim(pops):
    avgPops = np.mean(pops[:, :, 0], axis=1)
    grad = np.abs(np.gradient(avgPops, axis=0))
    maxGrad = np.max(grad)
    if maxGrad > 0.005:
        return True
    else:
        return False


# Collect Data
allData = {}
count = 0
for fold, _, files in os.walk(mroot_folder):
    for name in plotMe:
        numpy_files = any('.npy' in f for f in files)
        correct_folder_name = any("/"+nam+"/" in fold 
                                  for nam in folderNames[name])
        if correct_folder_name and numpy_files:
            m_model = re.findall("\/Model\_\d\/", fold)[0].strip("//").replace("Model_", "")
            m_model = int(m_model)
            if m_model == model:
                tmpData = {'data':{}}
                for fName in os.listdir(fold):
                    fPath = "%s/%s" % (fold, fName)
                    tmpData['data'][fName.replace(".npy", "")] = np.load(fPath)
                
                tmpData['model'] = int(m_model)
                tmpData['simType'] = name
                mom = re.findall("\/Kinit\_\d*", fold)[0].strip("//").replace("Kinit_", "")
                mom = int(mom) > 21
                tmpData['highMom'] = mom
                tmpData['folder'] = fold
                allData[count] = tmpData
                count += 1
            
for name in plotMe:
    print("\n"+name+":")
    allPopsLow, timeLow = [], []
    allPopsHigh, timeHigh = [], []
    badSims = []
    for i in allData:
        if allData[i]['simType'] == name:
            if is_bad_sim(allData[i]['data']['|C|^2']) * rm_bad_sims:
                badSims.append(i)
                continue
            if not allData[i]['highMom']:
                cond2 = len(allData[i]['data']['time']) > len(timeLow)
                if len(timeLow) == 0 or cond2:  # Only append once
                    timeLow = allData[i]['data']['time']
                allPopsLow.append(allData[i]['data']['|C|^2'])
            else:
                cond2 = len(allData[i]['data']['time']) > len(timeHigh)
                if len(timeHigh) == 0 or cond2:  # Only append once
                    timeHigh = allData[i]['data']['time']
                allPopsHigh.append(allData[i]['data']['|C|^2'])
        
    if len(allPopsLow) > 0:
        maxLowSteps = np.max([i.shape[0] for i in allPopsLow])
        allPopsLow = np.array([i for i in allPopsLow if i.shape[0] == maxLowSteps])
    if len(allPopsHigh) > 0:
        maxHighSteps = np.max([i.shape[0] for i in allPopsHigh])
        allPopsHigh = np.array([i for i in allPopsHigh if i.shape[0] == maxHighSteps])
#    allPopsHigh = np.array(allPopsHigh)
    print("Num Bad Sims = %i" % len(badSims))
    print("\tLow Sims = %i" % len(allPopsLow))
    print("\tHigh Sims = %i" % len(allPopsHigh))
    
    if 'Ener' in whichPlot:
        allPotLow, allKinLow = [], []
        allPotHigh, allKinHigh = [], []
        for i in allData:
            if allData[i]['simType'] == name:
                potE = np.sum(allData[i]['data']['E'] * allData[i]['data']['|C|^2'],
                              axis=2)
                if not allData[i]['highMom']:
                    allPotLow.append(potE)
                    allKinLow.append(1000 * (allData[i]['data']['vel'] ** 2))
                else:
                    allPotHigh.append(potE)
                    allKinHigh.append(1000 * (allData[i]['data']['vel'] ** 2))
    
        allPotLow, allKinLow = np.array(allPotLow), np.array(allKinLow)
        allPotHigh, allKinHigh = np.array(allPotHigh), np.array(allKinHigh)
        allTotLow, allTotHigh = allKinLow + allPotLow, allKinHigh + allPotHigh
    
    if 'normEner' in whichPlot:
        if len(allPopsLow) > 0:
            # Do the norm
            avgPopsLow = np.mean(np.sum(allPopsLow, axis=3), axis=2)  # avg over rep
            avgAvgPopsLow = np.mean(avgPopsLow, axis=0)
            stdAvgPopsLow = np.std(avgPopsLow, axis=0)
            ax[0, 0].plot(timeLow, 
                           avgAvgPopsLow, 'k-')
            ax[0, 0].plot(timeLow, 
                           avgAvgPopsLow + numStd * stdAvgPopsLow, 'k--')
            ax[0, 0].plot(timeLow, 
                           avgAvgPopsLow - numStd * stdAvgPopsLow, 'k--')

        if len(allPopsHigh) > 0:
            avgPopsHigh = np.mean(np.sum(allPopsHigh, axis=3), axis=2)  # avg over rep
            avgAvgPopsHigh = np.mean(avgPopsHigh, axis=0)
            stdAvgPopsHigh = np.std(avgPopsHigh, axis=0)
            ax[0, 1].plot(timeHigh, 
                          avgAvgPopsHigh, 'k-')
            ax[0, 1].plot(timeHigh, 
                          avgAvgPopsHigh + numStd * stdAvgPopsHigh, 'k--')
            ax[0, 1].plot(timeHigh, 
                          avgAvgPopsHigh - numStd * stdAvgPopsHigh, 'k--')
            
        # Do the energy conservation
        if len(allKinLow) > 0:
            avgKinLow = np.mean(allKinLow, axis=2)
            avgPotLow = np.mean(allPotLow, axis=2)
            avgTotLow = np.mean(allTotLow, axis=2)
            stdKL, stdPL = np.std(avgKinLow, axis=0), np.std(avgPotLow, axis=0)
            avgKL, avgTL = np.mean(avgKinLow, axis=0), np.mean(avgTotLow, axis=0)
            avgPL, stdTL = np.mean(avgPotLow, axis=0), np.std(avgTotLow, axis=0)
            ax[1, 0].plot(timeLow, avgKL, 'r-', label="Kin")
            ax[1, 0].plot(timeLow, 
                          avgKL + numStd * stdKL, 'r--')
            ax[1, 0].plot(timeLow, 
                          avgKL - numStd * stdKL, 'r--')
            ax[1, 0].plot(timeLow, avgPL, 'g-', label="Pot")
            ax[1, 0].plot(timeLow, 
                          avgPL + numStd * stdPL, 'g--')
            ax[1, 0].plot(timeLow, 
                          avgPL - numStd * stdPL, 'g--')
            ax[1, 0].plot(timeLow, avgTL, 'k-', label="Tot")
            ax[1, 0].plot(timeLow, 
                          avgTL + numStd * stdTL, 'k--')
            ax[1, 0].plot(timeLow, 
                          avgTL - numStd * stdTL, 'k--')

            Edrift, _ = np.polyfit(timeLow, avgTL, 1)
            Edrift *= 41341.3745758
            xPos = max(timeLow)*3./10.
            yPos = (max(avgTL) - min(avgTL)) + min(avgTL)
            ax[1, 0].annotate(r"Ener Drift = %.2g [$\frac{Ha}{atom \ ps}$]" % Edrift, (xPos, yPos),
                              fontsize=15)
            ax[1, 0].legend()

        if len(allKinHigh) > 0:
            avgKinHigh = np.mean(allKinHigh, axis=2)
            avgPotHigh = np.mean(allPotHigh, axis=2)
            avgTotHigh = np.mean(allTotHigh, axis=2)
            avgKH, stdTH = np.mean(avgKinHigh, axis=0), np.std(avgTotHigh, axis=0)
            stdKH, stdPH = np.std(avgKinHigh, axis=0), np.std(avgPotHigh, axis=0)
            avgPH, avgTH = np.mean(avgPotHigh, axis=0), np.mean(avgTotHigh, axis=0)
            
            ax[1, 1].plot(timeHigh, avgKH, 'r-', label="Kin")
            ax[1, 1].plot(timeHigh, 
                          avgKH + numStd * stdKH, 'r--')
            ax[1, 1].plot(timeHigh, 
                          avgKH - numStd * stdKH, 'r--')
            ax[1, 1].plot(timeHigh, avgPH, 'g-', label="Pot")
            ax[1, 1].plot(timeHigh, 
                          avgPH + numStd * stdPH, 'g--')
            ax[1, 1].plot(timeHigh, 
                          avgPH - numStd * stdPH, 'g--')
            ax[1, 1].plot(timeHigh, avgTH, 'k-', label="Tot")
            ax[1, 1].plot(timeHigh, 
                          avgTH + numStd * stdTH, 'k--')
            ax[1, 1].plot(timeHigh, 
                          avgTH - numStd * stdTH, 'k--')
            
            Edrift, _ = np.polyfit(timeHigh, avgTH, 1)
            Edrift *= 41341.3745758
            xPos = max(timeHigh)*3./10.
            yPos = (max(avgTH) - min(avgTH)) + min(avgTH)
            ax[1, 1].annotate(r"Ener Drift = %.2g [$\frac{Ha}{atom \ ps}$]" % Edrift, (xPos, yPos),
                              fontsize=15)
            ax[1, 1].legend()
        
        
        ax[1, 0].set_xlabel("Time [au]")
        ax[1, 1].set_xlabel("Time [au]")
        ax[0, 0].set_ylabel("Norm")
        ax[1, 0].set_ylabel("Energy [au]")
        ax[0, 0].set_title("Low Momentum", fontsize=18)
        ax[0, 1].set_title("High Momentum", fontsize=18)
    
    if 'compFred' in whichPlot:
        if len(allPopsLow):
            # Low Momentum
            decoLow = allPopsLow[:, :, :, 0] * allPopsLow[:, :, :, 1]
            avgDecoLow1 = np.mean(decoLow, axis=2)  # average over Replicas
            stdDecoLow = np.std(avgDecoLow1, axis=0)  # average over Replicas
            avgDecoLow = np.mean(avgDecoLow1, axis=0)  # average over Replicas
            
            avgPopsLow1 = np.mean(allPopsLow, axis=2) # average over replica first
            stdPopsLow = np.std(avgPopsLow1, axis=0)
            avgPopsLow = np.mean(avgPopsLow1, axis=0) # average over repeated runs
            
            if 'all' in std_or_allSim:
                for i in range(len(avgPopsLow1)):
                    pdAxes[0, 0].plot(timeLow, avgPopsLow1[i, :, 0], lw=0.3,
                                      color=colors[name])
            pdAxes[0, 0].plot(timeLow, avgPopsLow[:, 0], '-', color=colors[name])
            if 'std' in std_or_allSim:
                pdAxes[0, 0].plot(timeLow,
                                  avgPopsLow[:, 0]+numStd*stdPopsLow[:, 0],
                                  '--', color=colors[name])
                pdAxes[0, 0].plot(timeLow,
                                  avgPopsLow[:, 0]-numStd*stdPopsLow[:, 0],
                                  '--', color=colors[name])
            if 'all' in std_or_allSim:
                for i in range(len(avgDecoLow1)):
                    pdAxes[1, 0].plot(timeLow, avgDecoLow1[i], lw=0.3,
                                      color=colors[name])
            pdAxes[1, 0].plot(timeLow, avgDecoLow, '-', color=colors[name])
            if 'std' in std_or_allSim:
                pdAxes[1, 0].plot(timeLow, avgDecoLow-numStd*stdDecoLow, '--',
                                  color=colors[name])
                pdAxes[1, 0].plot(timeLow, avgDecoLow+numStd*stdDecoLow, '--',
                                  color=colors[name])
        
        # High Momentum
        if len(allPopsHigh):
#            allPopsHigh = 1 - allPopsHigh
            decoHigh = allPopsHigh[:, :, :, 0] * allPopsHigh[:, :, :, 1]
            avgDecoHigh1 = np.mean(decoHigh, axis=2)  # average over Replicas
            stdDecoHigh = np.std(avgDecoHigh1, axis=0)  # average over Replicas
            avgDecoHigh = np.mean(avgDecoHigh1, axis=0)  # average over Replicas
        
            avgPopsHigh1 = np.mean(allPopsHigh, axis=2) # average over replica first
            stdPopsHigh = np.std(avgPopsHigh1, axis=0)
            avgPopsHigh = np.mean(avgPopsHigh1, axis=0) # average over repeated runs
            
            if 'all' in std_or_allSim:
                for i in range(len(avgPopsHigh1)):
                    pdAxes[0, 1].plot(timeHigh, avgPopsHigh1[i, :, 0], lw=0.3,
                                      color=colors[name])
            pdAxes[0, 1].plot(timeHigh, avgPopsHigh[:, 0], '-',
                              color=colors[name])
            if 'std' in std_or_allSim:
                pdAxes[0, 1].plot(timeHigh,
                                  avgPopsHigh[:, 0]+numStd*stdPopsHigh[:, 0],
                                  '--', color=colors[name])
                pdAxes[0, 1].plot(timeHigh,
                                  avgPopsHigh[:, 0]-numStd*stdPopsHigh[:, 0],
                                  '--', color=colors[name])
        
            if 'all' in std_or_allSim:
                for i in range(len(avgDecoHigh1)):
                    pdAxes[1, 1].plot(timeHigh, avgDecoHigh1[i], lw=0.3,
                                      color=colors[name])
            pdAxes[1, 1].plot(timeHigh, avgDecoHigh, '-', color=colors[name])
            if 'std' in std_or_allSim:
                pdAxes[1, 1].plot(timeHigh,
                                  avgDecoHigh-numStd*stdDecoHigh, '--',
                                  color=colors[name])
                pdAxes[1, 1].plot(timeHigh,
                                  avgDecoHigh+numStd*stdDecoHigh, '--',
                                  color=colors[name])
    
    


        pdAxes[0, 0].set_title("Low Momentum", fontsize=18)
        pdAxes[0, 1].set_title("High Momentum", fontsize=18)

plt.savefig("Model_%i.png" % model)
plt.show()

#plt.close("all")



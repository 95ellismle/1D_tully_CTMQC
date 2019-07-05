from __future__ import print_function

import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os


def plotEcons(runData):
    """
    Will plot the conserved quantity along with the kinetice and potential
    energies.
    """
    lw = 0.1
    alpha = 0.5    
    
    f, a = plt.subplots()
    
    potE = np.sum(runData.allAdPop * runData.allE, axis=2)
    kinE = 0.5 * runData.ctmqc_env['mass'][0] * (runData.allv**2)
    totE = potE + kinE
    a.plot(runData.allt, potE, 'g', lw=lw, alpha=alpha)
    a.plot(runData.allt, kinE, 'r', lw=lw, alpha=alpha)
    a.plot(runData.allt, totE, 'k', lw=lw, alpha=alpha)
    
    avgTotE = np.mean(totE, axis=1)
    a.plot(runData.allt, np.mean(potE, axis=1), 'g', label="Pot. E")
    a.plot(runData.allt, np.mean(kinE, axis=1), 'r', label="Kin. E")
    a.plot(runData.allt, avgTotE, 'k', label="Tot. E")
    
    fit = np.polyfit(runData.allt, avgTotE, 1)
    annotateMsg = r"Energy Drift = %.2g" % (fit[0] * 41341.3745758)
    annotateMsg += r" [$\frac{Ha}{atom ps}$]"
    ypos = avgTotE[100] / 2.
    xpos = runData.allt[100]
    a.annotate(annotateMsg, (xpos, ypos), fontsize=20)

    print(r"Energy Drift = %.2g [Ha / (atom ps)]" % (fit[0] * 41341.3745758))
    
    a.legend()


def plotRlk_Rl(runData):
    """
    Will plot the Rlk and Rl term on the same axis
    """
    f, a = plt.subplots()
    a.plot(runData.allt, runData.allRl[:, 0], label=r"R$_{0}$")
    a.plot(runData.allt, runData.allRl[:, 1], label=r"R$_{1}$")
    a.plot(runData.allt, runData.allRlk[:, 0, 1], label="Rlk")
    a.plot(runData.allt, runData.allEffR, label=r"R$_{eff}$")
    a.set_xlabel("Time [au]")
    a.set_ylabel("Intercept [au]")
    a.legend()


def plotQlk(runData):
    lw = 0.1
    alpha = 0.5
    plt.figure()
    plt.plot(runData.allt, runData.allQlk[:, :, 0, 1], 'k', lw=lw, alpha=alpha)
    plt.plot(runData.allt, runData.allQlk[:, :, 0, 0], 'g', lw=lw, alpha=alpha)
    plt.plot(runData.allt, runData.allQlk[:, :, 1, 1], 'r', lw=lw, alpha=alpha)
    plt.plot(runData.allt, runData.allQlk[:, :, 1, 0], 'b', lw=lw, alpha=alpha)


def plotPops(runData):
    lw = 0.25
    alpha = 0.5
    f, a = plt.subplots()
    a.plot(runData.allt, runData.allAdPop[:, :, 1], 'b', lw=lw, alpha=alpha)
    a.plot(runData.allt, runData.allAdPop[:, :, 0], 'r', lw=lw, alpha=alpha)
    a.plot(runData.allt, np.mean(runData.allAdPop[:, :, 0], axis=1), 'r',
           label=r"|C$_{1}$|$^2$")
    a.plot(runData.allt, np.mean(runData.allAdPop[:, :, 1], axis=1), 'b',
           label=r"|C$_{2}$|$^2$")
    a.set_ylabel("Adiab. Pop.")
    a.set_ylabel("Time [au]")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])
    a.legend()


def plotNorm(runData):
    lw = 0.25
    alpha = 0.5
    allNorms = np.sum(runData.allAdPop, axis=2)
    avgNorms = np.mean(allNorms, axis=1)
    f, a = plt.subplots()
    a.plot(runData.allt, allNorms, 'r', lw=lw, alpha=alpha)
    a.plot(runData.allt, avgNorms, 'r')
    a.set_ylabel("Norm")
    a.set_ylabel("Time [au]")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])
    
    fit = np.polyfit(runData.allt, avgNorms, 1)
    print("Norm Drift = %.2g [$ps^{-1}$]" % (fit[0] * 41341.3745758))


def plotDeco(runData):
    lw = 0.1
    alpha = 0.5
    deco = runData.allAdPop[:, :, 0] * runData.allAdPop[:, :, 1]

    f, a = plt.subplots()
    a.plot(runData.allt, deco, 'k', lw=lw, alpha=alpha)
    a.plot(runData.allt, np.mean(deco, axis=1), 'k')
    a.set_ylabel("Coherence")
    a.set_ylabel("Time [au]")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])


def plotSigmal(runData):
    """
    Will plot sigmal against time
    """
    f, a = plt.subplots()
    a.plot(runData.allt, runData.allSigmal[:, 0], 'r', label=r"$\sigma_{1}$")
    a.plot(runData.allt, runData.allSigmal[:, 1], 'b', label=r"$\sigma_{2}$")
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$\sigma_{l}$")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])
    a.legend()


def plotPos(runData):
    """ 
    Will plot sigmal against time
    """
    lw = 0.3 
    alpha = 0.7 
    
    f, a = plt.subplots()
    a.plot(runData.allt, runData.allR, lw=lw, alpha=alpha)
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"$R^{(I)}$")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])


def plotNACV(runData):
    """ 
    Will plot NACV against position
    """
    lw = 1 
    alpha = 0.7 
    
    f, a = plt.subplots()
    a.plot(runData.allt, runData.allNACV[:, :, 0, 1], lw=lw, alpha=alpha)
    a.plot(runData.allt, runData.allNACV[:, :, 0, 0], lw=lw, alpha=alpha)
    a.plot(runData.allt, runData.allNACV[:, :, 1, 1], lw=lw, alpha=alpha)
    a.plot(runData.allt, runData.allNACV[:, :, 1, 0], lw=lw, alpha=alpha)
    a.set_xlabel("Time [au]")
    a.set_ylabel(r"NACV")
    a.set_title("%i Reps" % runData.ctmqc_env['nrep'])


def plot_single_Epot_frame(data, istep, saveFolder=False):
    """
    Will plot a single frame 
    """
    potE = np.sum(data['|C|^2'][istep] * data['E'][istep], axis=1)
    f, a = plt.subplots()
    a.plot(data['x'], data['E'][:, :, 0], 'r')
    a.plot(data['x'], data['E'][:, :, 1], 'b')
    a.plot(data['x'][istep], potE, 'k.')
    a.set_ylabel("Energy [Ha]")
    a.set_xlabel("Nucl. Pos [bohr]")
    if isinstance(saveFolder, str):
        f.savefig("%s/%06i.png" % (saveFolder, istep))
        plt.close()
    else:
        plt.show()

def plot_Epot_wrapper_func(args):
    """
    Acts as a wrapper to use the multiprocessing module.
    """
    if len(args) > 3:
        raise SystemExit("More arguments than is possible for the "
                         + "plot_single_Epot_frame function." +
                         "\n\t* Num Args Given: %i" % len(args) +
                         "\n\t* Max Args Allowed: 3")


    plot_single_Epot_frame(*args)


def orderFiles(folder, ext, badStr=False, replacer=False):
    """
    Will rename files to reorder them for ffmpeg.
    """
    # First find the files and sort them
    allFiles = [f for f in os.listdir(folder) if ext in f]
    if badStr is not False:
        allFiles = [f for f in allFiles if badStr not in f]
    if replacer is not False:
        allFiles = [f.replace(replacer, "") for f in allFiles]
    allNums = [int(f.replace(ext, "")) for f in allFiles]
    allFiles = [i[1] for i in sorted(zip(allNums, allFiles))]
    
    # Now change the names
    maxZeros = len(str(len(allFiles)))
    for i, oldF in enumerate(allFiles):
        newF = "0" * (maxZeros - len(str(i))) + str(i) + ".png"
        os.rename("%s/%s" % (folder, oldF), "%s/%s" % (folder, newF))
    
    ffmpegCmd = "ffmpeg -framerate 120 "
    ffmpegCmd += "-i %s/%%0%id.png potE_ani.mp4" % (folder, maxZeros)
    print("Use the following command to stitch the pics together:")
    print("\t`%s`" % ffmpegCmd)


def plotEpotTime(runData, which_steps=False, saveFolder=False):
    """
    Will plot the potential energy over time for either each step or just the
    one specified.
    """
    if isinstance(which_steps, int):
        data = {'x': runData.allR, 'E': runData.allE,
                '|C|^2': runData.allAdPop}
        plot_single_Epot_frame(data, which_steps, saveFolder)
    else:
        if which_steps is False:
            which_steps = range(0, runData.ctmqc_env['iter'])
        elif not isinstance(which_steps, (type(np.array(1)), list)):
            raise SystemExit("Sorry the argument `which_steps` has been"+
                             " inputted incorrectly. It should be an int, " +
                             "a list or a numpy array.")
        pool = multiprocessing.Pool()
        
        # First create the arguments to feed into the wrapper function
        data = []
        for istep in which_steps:
            d = {'x': runData.allR, 'E': runData.allE,
                 '|C|^2': runData.allAdPop}
            data.append(d)
        saveFolders = [saveFolder] * len(which_steps)
        
        # Feed in the arguments zipped up
        pool.map(plot_Epot_wrapper_func,
                            zip(data, which_steps, saveFolders))

        orderFiles(saveFolder, ".png")
        

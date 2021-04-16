import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import re

tConv = 1/41.34
globFontsize=24

def plot_my_data(data, fig_axes, momToPlot, lab="pops"):
    """
    Will plot the data that has been read from the simulation.

    Inputs:
        * data <dict> =>
        * fig_axes (plt.figure, np.array<plt.axis>) => Output of 'create_fig_ax'
        * momToPlot list<str> => The momenta to plot (give vals of momenta to plot in each graph)
        * lab <str, default 'pops'> => which quantity to plot'pops' or 'coherence'
    """
    f, axes = fig_axes

    data = data[lab]
    modCount=1
    for ix in range(2):
        for iy in range(2):
            ax = axes[ix, iy]
            modStr = f"Model{modCount}"

            # Get high or low momentum str
            momStr = momToPlot[modCount-1]
            if modStr not in data: continue
            mData = data[modStr]
            if momStr in mData: mData = mData[momStr]
            else: continue


            t, pops = mData
            t *= tConv
            if lab == 'pops':
                #all_fit = np.polyfit(t, pops[:,:,0] + pops[:,:,1], 1)
                ax.plot(t, np.mean(pops[:, :, 0], axis=1), color="k", lw=4, ls='--',
                        label="My Data")
            elif lab == "coherence":
                ax.plot(t, np.mean(pops, axis=1), color="k", lw=4, ls='--',
                        label="My Data")
            else:
                raise SystemExit(f"I don't know how to plot '{lab}'")
            #ax.plot(t, pops[:, :, 0], color="k", ls='--', lw=0.7, alpha=0.3)

            ax.legend(fontsize=globFontsize-1)
            modCount+=1


def plot_FredGoss_data(gossData, fredData, fig_axes, lab="pops",
                       mom='low', plot_vals=('Exact', 'CTMQC')):
    """
    Will plot Frederica and Gossel data on a set of axes.

    Inputs:
        * gossData <dict> => Output of 'read_FredGoss_data' (gossel folder)
        * fredData <dict> => Output of 'read_FredGoss_data' (frederica folder)
        * fig_axes (plt.figure, np.array<plt.axis>) => Output of 'create_fig_ax'
        * lab <str, default 'pops'> => which quantity to plot'pops' or 'coherence'
        * mom <str, default 'low'> => 'high' or 'low' momentum
        * plot_vals list<str>, default ['Exact', 'CTMQC'] => The quantities to plot

    Outputs: (list<str>)
        The momentums plotted
    """
    f, axes = fig_axes
    allMoms = []
    doFred = fredData is not False

    modCount=1
    for ix in range(2):
        for iy in range(2):
            ax = axes[ix, iy]

            modStr = f'Model{modCount}'

            # Get high or low momentum str
            momStrs = list(gossData[lab][modStr].keys())
            moms = [int(i.strip("K")) for i in momStrs]
            if mom == 'low':  momStr = momStrs[np.argmin(moms)]
            elif mom == 'high':  momStr = momStrs[np.argmax(moms)]
            else: raise SystemExit("Argument <mom> should be 'low' or 'high'")
            allMoms.append(momStr)

            # Point to correct data
            gD = gossData[lab][modStr][momStr]
            fD = False
            if doFred and momStr in fredData[lab][modStr]:
                fD = fredData[lab][modStr][momStr]

            # Plot
            for quant in plot_vals:
                x = gD[quant+"X"].dropna().astype(float)
                y = gD[quant+"Y"].dropna().astype(float)
                vals = sorted(zip(x, y))
                x, y = [i[0] for i in vals], [i[1] for i in vals]
                x, y = np.array(x), np.array(y)

                if quant == "Exact":
                    ax.plot(x*tConv, y, label=f"{quant} [Gossel, 18]",
                            lw=4, color="tab:green", ls='--')
                else:
                    ax.plot(x*tConv, y, label=f"{quant} [Gossel, 18]",
                            lw=4, ls='--')

            if fD is not False:
                for quant in plot_vals:
                    if quant == "Exact": continue
                    x = fD[quant+"X"].dropna().astype(float)
                    y = fD[quant+"Y"].dropna().astype(float)
                    vals = sorted(zip(x, y))
                    x, y = [i[0] for i in vals], [i[1] for i in vals]
                    x, y = np.array(x), np.array(y)

                    ax.plot(x*tConv, y, label=f"{quant} [Agostini, 16]", ls='--', lw=4)


            modCount+=1
            #ax.legend(fontsize=globFontsize)

    return allMoms

def create_fig_ax(ylabel):
    """
    Will create the figure and axis for the plots.

    The figure will have 4 subplots (arranged in an even grid).

    Inputs:
        * ylabel <str> => The label for the yaxis
                           (e.g. "$|C|^2$" or "coherence")

    Outputs  (plt.figure, np.array<plt.axis>):
        A tuple (2 elements) with figure and axis. Axis is a np.array 2x2.
    """
    f, a = plt.subplots(2,2, figsize=(20, 10))

    # Set xlabel
    for ax in a[-1, :]:
        ax.set_xlabel("Timestep [fs]", fontsize=globFontsize)

    for ax in a[:, 0]:
        ax.set_ylabel(ylabel, fontsize=globFontsize)

    modCount = 1
    for ix in range(2):
        for iy in range(2):
            a[ix, iy].set_title(f"Model {modCount}", fontsize=globFontsize)
            modCount += 1

    return f, a

def _read_numpy(fold, lab):
    """
    Will read the output of my simulations

    Inputs:
        * fold <str> => The folder to find data in
        * lab <str> => 'pops' or 'coherence'
    """
    fp = f"{fold}/|C|^2.npy"
    fpT = f"{fold}/time.npy"
    if not os.path.isfile(fp): return False

    pops = np.load(fp)
    time = np.load(fpT)
    if lab == "pops": return time, pops
    if lab == "coherence": return time, pops[:, :, 0] * pops[:, :, 1]

def _read_csv(fold, lab):
    """
    Will read a csv data file contianing gossel or fred data

    Inputs:
        * fold <str> => The folder to find data in
        * lab <str> => 'pops' or 'coherence'
    """
    fp = f"{fold}/{lab}.csv"
    if not os.path.isfile(fp): return False

    df = pd.read_csv(fp)
    # Fix columns for Gossel Data
    if len(df.columns) == 6:
        df.columns = ("ExactX", "ExactY", "EhrenX", "EhrenY", "CTMQCX", "CTMQCY")
    # Fix columns for Frederica data
    if len(df.columns) == 10:
        df.columns = ("ExactX", "ExactY", 'SHX', 'SHY', "EhrenX", "EhrenY", 'MQCX', 'MQCY', "CTMQCX", "CTMQCY")

    df = df.iloc[1:]
    return df


def read_data(folder, func=_read_csv):
    """
    Will read all the data from either the Frederica/Gossel Data folder.

    Folder struct = Data/Model?/CTMQC_??K/coherence.csv or pops.csv

    Inputs:
        * folder <str> => The folder to look in for the data
        * func <function, default _read_csv> => The data read function

    Outputs: (<dict>)
        The data in a nested dictionary one with pops and coherences.
            Dict structure = <'pops'|'coherence'><'Model?'><'??K'>
    """
    if not os.path.isdir(folder):
        raise SystemExit(f"Can't find data folder: '{folder}'")

    data = {'pops': {}, 'coherence': {}}

    defaultFunc = lambda txt: re.findall('[0-9]+', txt)[0]
    mapDict = {'model.*': (defaultFunc, 'Model.'),
               'kinit.*': (defaultFunc, '.K'),
               'ctmqc_[0-9]+k': (defaultFunc, '.K'),
               'sig_[0-9.]+': (lambda txt: re.findall('[0-9.]+', txt)[0], 'Sigma_.'),
              }

    for folder, folders, files in os.walk(folder):
        if not len(files): continue

        for lab in ('pops', 'coherence'):

            # Create the nested structure
            splitter = folder.split('/')
            currentStruct = {}
            tmp = data[lab]
            tmp2 = currentStruct
            for fold in splitter:
                for poss in mapDict:
                    if re.search(poss, fold.lower()):
                        name = mapDict[poss][1].replace('.', mapDict[poss][0](fold))
                        tmp.setdefault(name, {})
                        tmp2.setdefault(name, {})
                        tmp = tmp[name]
                        tmp2 = tmp2[name]

            # Pop the data in the correct place
            tmp = data[lab]
            tmp2 = currentStruct
            while (len(tmp2.keys()) > 0):
                key = list(tmp2.keys())[0]

                if len(tmp2[key].keys()) == 0:
                    df = func(folder, lab)
                    if df is False: continue
                    tmp[key] = df
                    break

                tmp = tmp[key]
                tmp2 = tmp2[key]

    return data

#    for modelF in os.listdir(folder):
#        fold = f"{folder}/{modelF}"
#        if os.path.isfile(fold): continue
#        modStr = "Model" + re.findall("[0-9]+", modelF)[0]
#
#        for momF in os.listdir(fold):
#            fold = f"{folder}/{modelF}/{momF}"
#            if os.path.isfile(folder): continue
#
#            momF = re.findall("[0-9]+", momF)[0] + 'K'
#            for lab in ('pops', 'coherence'):
#
#                df = func(fold, lab)
#                if df is False: continue
#
#                data[lab].setdefault(modStr, {})  \
#                         .setdefault(momF, df)
#
    return data

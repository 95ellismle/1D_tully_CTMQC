import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

MOMENTA = ['high']
MODELS = [1, 2, 3, 4]
QUANTITIES = ['pops', 'coherence']
CT_or_EHs = ['Ehrenfest']
root_folder = "/scratch/mellis/TullyModelData/FullCTMQCGossel"



def tidyFile(filename):
    """
    Will merge the 2 header rows into 1 row.
    """
    with open(filename, 'r') as f:
        txt = f.read().split("\n")
        names = txt[0]
        xy = txt[1]
        newNames = []
        for name, XY in zip(names.split(','), xy.split(',')):
            if name:
                tmp = name
            newName = tmp + "_" + XY
            newNames.append(newName.strip())
        newNames = ','.join(newNames)
    
    txt = [newNames] + txt[2:]
    txt = [i for i in txt if i]
    
    with open("tmp.csv", 'w') as f:
        f.write('\n'.join(txt))


def tidyData(df):
    """
    Will do various things to tidy the data in the datafram (such as sort by x,
    ammend any unrealistic values such as time < 0, 1 < population < 0 ...
    """
    cols = set([col.replace('_X', '').replace('_Y', '').strip()
                for col in df.columns])
    
    for col in cols:
        xCol, yCol = "%s_X" % col, "%s_Y" % col
        sortedAll = sorted(zip(df[xCol], df[yCol]))
        sortedY = [i[1] for i in sortedAll]
        sortedX = [i[0] for i in sortedAll]
        df[xCol] = sortedX
        df[yCol] = sortedY
    
        df[yCol][df[yCol] > 1] = 1
        df[yCol][df[yCol] < 0] = 0
        df[xCol][df[xCol] < 0] = 0
    
    return df
    

def create_filename(root_folder, model, momentum, which_quantity):
     """
     Will create the filename that corresponds to the input variables.
     """
     filename = '%s/Model_%i/%sMom/%s.csv' % (root_folder, model,
                                              momentum.title(),
                                              which_quantity)
     return filename


def read_csv(root_folder, model, momentum, which_quantity):
    """
    Will read a csv file and tidy up the columns etc...
    """
    filename = create_filename(root_folder, model, momentum, which_quantity)
    tidyFile(filename)
    
    df = pd.read_csv("tmp.csv")
    df = tidyData(df)
        
    return df


def plot_Gossel_data(params):
    """
    Will plot the data from the Gossel paper.
    
    Inputs:
        params = dict with params of what to plot. Keys are:

                Required:
                * rootFolder = the base folder for the data
                * tullyModel = the model that is to be plotted (1, 2, 3, 4)
                * momentum = high or low momentum ('high', 'low')
                * whichQuantity = which thing to plot ('nuclDens', 'pops',                                     
                                                       'coherence')
                * whichSimType = what type of simulation data ("CTMQC",
                                                               "Ehrenfest",
                                                               "Exact")
                * colors = the color of the plotted line (dict with
                                                          whichSimType as a
                                                          keys)
                Optional:
                * xlabel = the xlabel for the graph
                * ylabel = the ylabel for the graph
                * title = the title of the graph
                * lw = linewidth of the line plotted
    """
    momentums = {1: {'high': 25, 'low': 15},
                 2: {'high': 30, 'low': 16},
                 3: {'high': 30, 'low': 10},
                 4: {'high': 40, 'low': 10}}
    root_folder, model = params['rootFolder'], params['tullyModel']
    momentum, which = params['momentum'], params['whichQuantity']
    if not isinstance(momentum, (str)):
         if momentum:
             momentum = 'High'
         else:
             momentum = 'Low'
    
    df = read_csv(root_folder, model, momentum, which)
    
    f, a = plt.subplots()
    a.set_xlabel(params.get('xlabel', ""))
    a.set_ylabel(params.get('ylabel', ""))
    mom = momentums[model][momentum.lower()]
    default_title = r"Model %i: $\mathbf{P}_{init}$ = %i au" % (model, mom)
    a.set_title(params.get('title',default_title))
    
    plotParams = params['whichSimType']
    if not isinstance(plotParams, list):
        plotParams = [plotParams]
    for plotParam in plotParams:
        a.plot(df[plotParam+'_X'], df[plotParam+'_Y'],
               color=params['colors'][plotParam], lw=params.get('lw', 2),
               alpha=params.get('alpha', 0.8),
               label=plotParam + " (Gossel Paper)")
    
    a.legend()
    return f, a


def get_params_from_folder(folderpath, params={}):
    """
    Will get the parameters of the run from the tullyInfo.npy file in the
    folder. If not will try to extract them from the name.
    """
    filepath = "%s/tullyInfo.npy" % os.path.abspath(folderpath)
    if os.path.isfile(filepath):
        runParams = np.load(filepath)
        runParams = eval(str(runParams))
        for key in runParams:
            params[key] = runParams[key]
        if 'mass' not in runParams:
            params['mass'] = 2000
        mom = params['mass'] * params['velInit']
        params['momentum'] = 'low'
        if mom > 20:
            params['momentum'] = 'high'
            
    return params


def load_all_data(root_folder):
    """
    Will load all the data from a root folder by walking across the available
    files and loading the data from each one.
    """
    rootFolder = os.path.abspath(root_folder)
    metadata = {'model':[], 'momentum':[], 'repeat':[], 'name':[],
                'data_inds':[], 'CTMQC': []}
    data = []
    uniqueIdentifiers = []
    count = 0
    for path, folders, files in os.walk(rootFolder):
        if "|C|^2.npy" in files:
            for f in files:
                if '.npy' in f:
                    params = get_params_from_folder(path)
                    
                    dataHeader = f.replace(".npy", "")
                    model = params['tullyModel']
                    mom = params['momentum']
                    uniqueIdentifier = "%s_%s_%s" % (dataHeader, model, mom)
                    uniqueIdentifiers.append(uniqueIdentifier)
                    repeat = uniqueIdentifiers.count(uniqueIdentifier)
                    ehrenCTMQC = params['do_QM_C'] * params['do_QM_F']
                    
                    metadata['model'].append(model)
                    metadata['momentum'].append(mom)
                    metadata['repeat'].append(repeat)
                    metadata['name'].append(dataHeader)
                    metadata['data_inds'].append(count)
                    metadata['CTMQC'].append(ehrenCTMQC)
                    data.append(np.load("%s/%s" % (path, f)))
                    count += 1
    metadata = pd.DataFrame(metadata)
    data = np.array(data)
    
    return data, metadata


def get_data_from_array(data, metadata, masks):
    """
    Will get some data from the allData dataframe using the masks provided in
    the input dictionary
    """
    mask = np.all([metadata[key] == masks[key] for key in masks], axis=0)
    pointers = metadata[mask]
    return np.array([i for i in data[pointers['data_inds']]])


def plot_my_pops(data, metadata, model, momentum, CT_or_EH, f, a):
    """
    Will plot the population data from my simulations
    """
    masks = {'model': model, 'name': 'time', 'momentum': momentum,
             'CTMQC': CT_or_EH}
    time = get_data_from_array(data, metadata, masks)
    masks = {'model': model, 'name': '|C|^2', 'momentum': momentum,
             'CTMQC': CT_or_EH}
    pops = get_data_from_array(data, metadata, masks)
    
    time = np.mean(time, axis=0)
    pops = np.mean(pops, axis=2)  # average over replicas
    stdPops = np.std(pops, axis=0)  # stddev over repeats
    pops = np.mean(pops, axis=0)  # average over repeats
    
    a.plot(time, pops[:, 0], 'k', label="My Data")
    a.plot(time, pops[:, 0] - stdPops[:, 0], 'k--')
    a.plot(time, pops[:, 0] + stdPops[:, 0], 'k--')
    a.legend()


def plot_my_coherence(data, metadata, model, momentum, CT_or_EH, f, a):
    """
    Will plot the population data from my simulations
    """
    masks = {'model': model, 'name': 'time', 'momentum': momentum,
             'CTMQC': CT_or_EH}
    time = get_data_from_array(data, metadata, masks)
    masks = {'model': model, 'name': '|C|^2', 'momentum': momentum,
             'CTMQC': CT_or_EH}
    pops = get_data_from_array(data, metadata, masks)
    
    time = np.mean(time, axis=0)
    coherence = pops[:, :, :, 0] * pops[:, :, :, 1]
    coherence = np.mean(coherence, axis=2)  # average over replicas
    stdCoh = np.std(coherence, axis=0)  # stddev over repeats
    coherence = np.mean(coherence, axis=0)  # average over repeats
    
    a.plot(time, coherence, 'k', label="My Data")
    a.plot(time, coherence - stdCoh, 'k--')
    a.plot(time, coherence + stdCoh, 'k--')
    a.legend()


data, metadata = load_all_data(root_folder)
for MOMENTUM in MOMENTA:
    for MODEL in MODELS:
        for QUANTITY in QUANTITIES:
            for CT_or_EH in CT_or_EHs:
                params = {'rootFolder': '/homes/mellis/Documents/Graphs/' +
                                        'Tully_Models/GosselPaper/Data',
                          'momentum': MOMENTUM,
                          'tullyModel': MODEL,
                          'whichQuantity': QUANTITY,
                          'whichSimType': [CT_or_EH],  # , 'Ehrenfest', 'Exact'
                          'colors': {'Ehrenfest': 'b',
                                     'CTMQC': 'r',
                                     'Exact': 'k'},
                          'xlabel': 'Time [au]',
                          'ylabel': 'Adiabatic Populations',
                          'lw': 3,
                          'alpha': 0.7,
                          }
                
                f, a = plot_Gossel_data(params)
                myCTMQC = CT_or_EH == "CTMQC"
                if QUANTITY == 'pops':
                    plot_my_pops(data, metadata, MODEL, MOMENTUM,
                                 myCTMQC, f, a)
                elif QUANTITY == 'coherence':
                    plot_my_coherence(data, metadata, MODEL, MOMENTUM,
                                      myCTMQC, f, a)
                
                save_root = "/homes/mellis/Documents/Graphs/Tully_Models/" + \
                            "GosselPaper"
                save_folderpath = "%s/%s/Model%i/%sMom" % (save_root,
                                                           CT_or_EH,
                                                           MODEL,
                                                           MOMENTUM.title())
                if not os.path.isdir(save_folderpath):
                    os.makedirs(save_folderpath)
                
                savepath = "%s/%s.png" % (save_folderpath, QUANTITY)
                f.savefig(savepath)
                plt.close('all')
            

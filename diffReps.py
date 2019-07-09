import os
import numpy as np
import matplotlib.pyplot as plt

folders = 'MomentumRuns'


def read_all_data_in_folders(folderPaths):
    """
    Will read all the data in a list of folderPaths
    """
    if isinstance(folderPaths, str):
        folderPaths = [folderPaths]
    elif not isinstance(folderPaths, list):
        raise SystemExit("Wrong type for input argument of read_all_data." + 
                         "\nShould be <str> or <list>.")

    allData = {}
    count = 0
    for f in folderPaths:
        if os.path.isdir(f):
            print(f)
            for folder, folders, files in os.walk(f):
                if any('.npy' in j for j in files):
                    allData[count] = {}
                    folderPath = os.path.join(f, folder)
                    data = {}
                    for dataFile in files:
                        dataName = dataFile.replace(".npy", "")
                        dataFilePath = os.path.join(folder, dataFile)
                        allData[count][dataName] = np.load(dataFilePath)

                    allData[count]['folderpath'] = folderPath
                    count += 1

    return allData


def get_E_prop(data):
    """
    Will get the kinetic, potential and total energy
    """
    E = data['E']
    pops = data['|C|^2']
    vel = data['vel']

    potE = np.sum(pops * E, axis=2)
    kinE = 0.5 * 2000 * vel**2
    totE = potE + kinE

    return {'tot': totE,
            'kin': kinE,
            'pot': potE,
            't': data['time']}


def plot_data_keys(data, xkey, ykey):
    f, a = plt.subplots()
    y_data = data[ykey]
    lw, alpha = 0.5, 0.5
    nrep = data['|C|^2'].shape[1]

    if len(y_data.shape) == 3:
        colors = ['r', 'b', 'g']
        for i in range(y_data.shape[2]):
            a.plot(data[xkey], y_data[:, :, i], color=colors[i],
                   lw=lw, alpha=alpha)
    elif len(y_data.shape) == 4:
        colors = ['r', 'g', 'b', 'k']
        c = 0
        for i in range(y_data.shape[2]):
            for j in range(y_data.shape[3]):
                a.plot(data[xkey], y_data[:, :, i, j], color=colors[c],
                       lw=lw, alpha=alpha)
                c += 1
    elif len(y_data.shape) == 2:
        a.plot(data[xkey], y_data, 'k', lw=lw, alpha=alpha)
    else:
        raise SystemExit("Can't plot that dimension data yet")

    a.set_xlabel(r"%s" % xkey.title())
    a.set_ylabel(r"$%s$" % ykey.title())
    a.set_title("Nrep = %i" % nrep)
    return f, a

def plotDiffMomData(allData, tullyModel): 
    modelData = apply_to_all_data(allData, lambda x: x,
                                  filters=("tullyModel", tullyModel))

    allVel = [modelData[i]['vel'][0] for i in modelData]
    allPops = [modelData[i]['|C|^2'][-1] for i in modelData]
    allVel = np.array(allVel)[:, 0]
    allPops = np.array(allPops)[:, 0, :]

    f, a = plt.subplots() 
    a.plot(allVel * 2000., allPops[:, 0], 'k.', label="Ehrenfest") 
    a.set_xlabel("Intial Momentum [au]")  
    a.set_ylabel("Final Ground State Population") 
    a.legend() 
    a.set_title("Model %s" % tullyModel) 
    return f, a


def apply_to_all_data(allData, func, args=[], filters=False):
    """
    Will apply a function to every data key in the allData dictionary.
    """
    if filters is False:
        return {i: func(allData[i], *args)
                  for i in allData}
    else:
        returner = {}
        for count, ind in enumerate(allData):
            D = eval(str(allData[ind]['tullyInfo']))
            if str(D[filters[0]]) == str(filters[1]):
                returner[count] = func(allData[ind], *args)
        return returner


allData = read_all_data_in_folders(folders)

f, a = plotDiffMomData(allData, 4)


#allEprops = apply_to_all_data(allData, get_E_prop)
##apply_to_all_data(allData, plot_data_keys, ('time', '|C|^2'))
#apply_to_all_data(allData, plot_data_keys, ('time', 'Qlk'))
#class classify(object): 
#    def __init__(self, data): 
#        self.allR = data['pos'] 
#        self.allE = data['E'] 
#        self.allAdPop = data['|C|^2'] 
#        self.allt = data['time']
#plt.show()

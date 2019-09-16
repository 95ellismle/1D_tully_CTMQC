#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:50:40 2019

@author: mellis
"""

import os
import numpy as np
import pandas as pd

FredDataFold = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/FredericaData"
GossDataFold = "/scratch/mellis/TullyModelData/Big_ThesisChap_Test/GosselData"

def check_folder(folderpath):
    """
    Check for folder existance
    """
    folderpath = os.path.abspath(folderpath)
    if not os.path.isdir(folderpath):
        print("Can't find folder `%s`" % folderpath)
        return False
    return folderpath


def add_to_list_in_dict(D, key, val):
        """
        Will add a value to a list within a dictionary if the key doesn't
        exist. If it does then this will simply append to that list.
        """
        if key in D:
            D[key].append(val)
        else:
            D[key] = [val]


class SingleSimData(object):
    """
    Will store data from a single trajectory.
    """
    # To convert the params to allow the user a bit more flex with their input
    conv_input_to_filename = {'|C|^2': ['ad pops'],
                 'Feh': ['ehren force'],  'E': ['ener'],
                 'H': ['ham'], 'pos': ['R'], 'Fqm': ['qm force'],
                 'RI0': ['alt intercept', "R_0", "R0"], 'sigmal': ['sigma_l'],
                 'time': ['T'], 'u': ['diab coeff'], 'C': ['ad coeff'],
                 'Fad': ['ad force'], 'f': ['ad mom'], 'NACV': ['dlk'],
                 'Qlk': ['QM', 'quantum momentum'], 'Rlk': ['intercept'],
                 'sigma':[], 'vel': ['v'], 'Ftot': ['tot force']}

    # What the filenames will be saved as
    conv_file_to_param_names = {'|C|^2': 'adPop', 'E': 'E', 'Feh': 'Feh',
                 'H': 'H', 'pos': 'R', 'Fqm': 'Fqm', 'RI0': 'RI0',
                 'sigmal': 'sig_l', 'time': 'times', 'u': 'u', 'C': 'C',
                 'Fad': 'Fad', 'f': 'f', 'NACV': 'dlk', 'Qlk': 'Qlk',
                 'Rlk': 'Rlk', 'sigma':'sig', 'vel': 'v', 'Ftot': 'F'}
    def __init__(self, folderpath, params_to_read=False):

        # Check if the folder exists
        self.folderpath = check_folder(folderpath)
        if not self.folderpath: return None

        # Store the tully info as params
        self._store_tully_data()

        # Read any params that have been requested
        if params_to_read:
            self._get_all_data(params_to_read)

    def _store_tully_data(self):
        """
        Will read the data from the tullyInfo.npy file within the folder and
        store it as params in the class so they're easy to use.
        """
        dataFilepath = "%s/tullyInfo.npy" % self.folderpath
        if not os.path.isfile(dataFilepath):
            print("Can't find the `tullyInfo.npy` file")
            return False

        data = np.load(dataFilepath)
        data = eval(str(data))
        for key in data:
            # Try to remove the rounding errors
            if type(data[key]) == float:
                data[key] = round(data[key], 5)
            setattr(self, key, data[key])

    def _get_all_data(self, params_to_read):
        """
        Will get all the requested data from the `params_to_read` variable.
        """
        # If a single param is given outside a list
        if type(params_to_read) == str:
            if params_to_read == 'all':
                params_to_read = [f.replace('.npy', '') for f in os.listdir(self.folderpath)
                                  if '.npy' in f and 'tullyInfo' not in f]
            else:
                params_to_read = [params_to_read]

        # Convert the param names to filenames
        for param in params_to_read:
            poss_params = []
            for key in self.conv_file_to_param_names:
                if key.lower() == param.lower():
                    poss_params.append(key)
                    continue

                for val in self.conv_input_to_filename[key]:
                    if val.lower() == param.lower():
                        poss_params.append(key)
                            
            if len(poss_params) != 1:
                print("I don't know how to handle the parameter name `%s`." % param)
                continue

            filename = "%s.npy" % poss_params[0]
            filepath = os.path.join(self.folderpath, filename)
            if not os.path.isfile(filepath):
                print("I can't find the file `%s`." % filename)
                print("This is probably an error with the `conv_input_to_filename` in the class")
                continue

            data = np.load(filepath)
            setattr(self, self.conv_file_to_param_names[poss_params[0]], data)

    def _check_necessary_quantities(self, necessary_quants):
        """
        Will check that the quantities given in the list or tuple input can be found in the class
        """
        if all(j in self.__dict__.keys() for j in necessary_quants): return True
        for quant in necessary_quants:
            if quant not in self.__dict__.keys():
                print("Sorry I haven't loaded the `%s`" % quant)
                print("Please change the `params_to_load` input to load this")
        return False

    def get_norm_drift(self):
        """
        Will get the data from the adiabatic populations
        """
        necessary_quants = ('times', 'adPop')
        check = self._check_necessary_quantities(necessary_quants)
        if not check: return False

        norm = np.sum(self.adPop, axis=2)
        norm = np.mean(norm, axis=1)

        fit = np.polyfit(self.times, norm, 1)
        slope = fit[0]
        slope *= 41341.373335182114

        return slope


    def get_ener_drift(self):
        """
        Will get the data from the adiabatic populations
        """
        necessary_quants = ('times', 'E', "adPop", "v")
        check = self._check_necessary_quantities(necessary_quants)
        if not check: return False

        Epot = np.sum(self.adPop * self.E, axis=2)
        Ekin = 1000 * (self.v**2)
        Etot = np.mean(Ekin + Epot, axis=1)

        fit = np.polyfit(self.times, Etot, 1)
        slope = fit[0] * 41341.3745758

        return slope

    def get_coherence(self):
        """
        Will get the coherence when asked (if the adiabatic populations have been loaded).
        """
        necessary_quants = ('times', 'adPop')
        check = self._check_necessary_quantities(necessary_quants)
        if not check: return False

        coherence = self.adPop[:, 0] * self.adPop[:, 1]
        return coherence


class NestedSimData(object):
    """
    Will read the data from many simulations that are in nested folders. To
    acess the data one can use the function `query_data`. The data is stored in
    a flat list and a map is created that can be queried.
    """
    def __init__(self, folderpath, params_to_read=False):

        # Check the folder exists
        self.folderpath = check_folder(folderpath)
        if not self.folderpath: return None

        # Get all the data and store it in a list
        self._get_data(params_to_read)

    def _get_data(self, params_to_read):
        """
        Will read the data from many simulations that are nested and return a dict
        that has the same nested structure as the folders.
        """

        self.allData = []
        self.__allDataMap = {}
        count = 0
        for fold, folders, files in os.walk(self.folderpath):
            if any('.npy' in i for i in files):
                # Save the data
                trajData = SingleSimData(fold, params_to_read)
                self.allData.append(trajData)
                add_to_list_in_dict(self.__allDataMap, "index", count)

                # Now create the map to query the data
                for key, value in trajData.__dict__.items():
                    if type(value) is not list:
                        if key == 'dt':
                            value = round(value, 3)
                        add_to_list_in_dict(self.__allDataMap, key, value)

                count += 1
        self.__allDataMap = pd.DataFrame(self.__allDataMap)

    def query_data(self, query_dict):
        """
        Will allow the easy searching of the data in the nested folders.

        Inputs:
            * query_dict = a dictionary in the format: {'quantity': 'value'}

                           In this case the quantity will be one of the values
                           in the tullyInfo file (the attrs of the main class
                           in the simulation e.g. elec_steps).

                           The value will be the value of the quantity.

                           The keys will act via 'and' and values within keys
                           will act via 'or'.

                           E.g:
                           If one only wanted data with nuclear dt of 0.1 then
                           a query would be:
                               {'dt': 0.5}.

                           If one wanted a nuclear
                           dt of [0.1, 0.2 or 0.5] then the query would be:
                               {'dt': [0.1, 0.2, 0.5]}.

                           If one wanted tully model 1 or 2 data which also
                           had a timestep of 0.1 then the query would be:
                               {'dt':0.1, 'tullyModel': [1, 2]}

        NOTE: When querying the nuclear timestep use a number rounded to 3 d.p.
        """
        # First check if the queries are valid
        remove_from_dict = []
        for query in query_dict:
            if query not in self.__allDataMap.columns:
                print("I can't find the query string `%s` in the data map." % query)
                print("Available query quantities are:\n\t*%s" % "\n\t*".join(self.__allDataMap.columns))
                remove_from_dict.append(query)
                continue

            if type(query_dict[query]) is not list:
                query_dict[query] = [query_dict[query]]

        # Remove invalid queries
        for rem in remove_from_dict:
            query_dict.pop(rem)

        # Now apply the queries
        mask = np.ones(len(self.__allDataMap), dtype=bool)
        for query in query_dict:
            newMask = np.zeros(len(self.__allDataMap), dtype=bool)
            for val in query_dict[query]:
                newMask = newMask | (self.__allDataMap[query] == val)

            mask = mask & newMask
        allInds = self.__allDataMap['index'][mask]
        return [self.allData[i] for i in allInds]


class FredericaData(object):
    """
    Will load Frederica's data and save each simulation/model as an attribute
    in the class.

    Inputs:
        * folderpath = root of the place the data is stored (1 above the Model
                       folders).
    """
    names = ['exact', 'SH', 'Eh', 'MQC', 'CTMQC']
    deco_filename = "Deco"
    pops_filename = "Pops"
    refStr = "Agostini"
    def __init__(self, folderpath=FredDataFold):
        # Check if the folderpath exists
        self.folderpath = check_folder(folderpath)
        if not self.folderpath: raise SystemExit("Can't find %s Data" % self.refStr)

        self._read_all_data()

    def __read_indivual_files(self, filepath):
        """
        Will read an individual csv file and tidy up the data.
        """
        headers = [name + j for name in self.names for j in ['_x', '_y']]
        df = pd.read_csv(filepath, skiprows=[0, 1], names=headers)
        df = self.__tidy_data(df)

        return df

    def __tidy_data(self, df):
        """
        Will tidy up the data by imposing some physical constraints on it
        """
        # Make sure there is no data above 1 or below 0
        for name in self.names:
            yname = name+'_y'
            df[yname][df[yname] > 1] = 1
            df[yname][df[yname] < 0] = 0

        # Now sort the data by the y value
        for name in self.names:
            xname, yname = name+'_x', name+'_y'
            xdata, ydata = df[xname], df[yname]

            sortedData = sorted(zip(xdata, ydata))
            xdata = np.array([i[0] for i in sortedData])
            ydata = np.array([i[1] for i in sortedData])

            df[xname] = xdata
            df[yname] = ydata

        return df

    def find_high_low_mom(self, allMoms, model):
        """
        Will determine which momentum is high or low
        """
        momentums = [int(i.split("_")[1].strip('K'))
                     for i in allMoms]
        allIsHigh = [i == max(momentums) for i in momentums]

        setattr(self, "mod%i_highMom" % model, max(momentums))
        setattr(self, "mod%i_lowMom" % model, min(momentums))

        return allIsHigh

    def _read_all_data(self):
        """
        Will read all the data in the folders.
        """
        # Get all the folders for each model
        for model in range(1, 5):
            newFolderO = "%s/Model%i" % (self.folderpath, model)
            newFolderO = check_folder(newFolderO)
            if not newFolderO: raise SystemExit("Can't find %s Data, model %i" % (self.refStr, model))

            allIsHigh = self.find_high_low_mom(os.listdir(newFolderO), model)
            for fold, isHigh in zip(os.listdir(newFolderO), allIsHigh):
                newFolder = "%s/%s" % (newFolderO, fold)
                decoF = "%s/%s.csv" % (newFolder, self.deco_filename)
                popsF = "%s/%s.csv" % (newFolder, self.pops_filename)

                momStr = "low"
                if isHigh: momStr = 'high'

                attr = "mod%i_%sMom_deco" % (model, momStr)
                setattr(self, attr, self.__read_indivual_files(decoF))
                attr = "mod%i_%sMom_pops" % (model, momStr)
                setattr(self, attr, self.__read_indivual_files(popsF))



class GosselData(FredericaData):
    """
    Will load Gossel's data and save each simulation/model as an attribute
    in the class.

    Inputs:
        * folderpath = root of the place the data is stored (1 above the Model
                       folders).
    """
    names = ['exact', 'Eh', 'CTMQC']
    deco_filename = "coherence"
    pops_filename = "pops"
    refStr = "Gossel"
    def __init__(self, folderpath=GossDataFold):
        # Check if the folderpath exists
        self.folderpath = check_folder(folderpath)
        if not self.folderpath: raise SystemExit("Can't find %s Data" % self.refStr)

        self._read_all_data()

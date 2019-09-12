#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:05:11 2019

@author: mellis
"""

import matplotlib.pyplot as plt

import numpy as np

def plotNormVsElecDt(allNestedData, model, axis=False):
    """
    Will plot the electronic timestep vs the norm for all the data in the
    nested data input for a certain model. BOB
    """ 
    allData = allNestedData.query_data({'tullyModel': model})
    dt = allData[0].dt * 24.188843265857  # in as
    elec_dts = [dt / data.elec_steps for data in allData]
    norms = [abs(data.get_norm_drift()) for data in allData]
    
    if axis is False:
        f, axis = plt.subplots()
    
    axis.plot(elec_dts, norms, 'ko', ls='--')
    axis.set_yscale("log")
    axis.grid(color="#dddddd")
    axis.set_ylabel("Norm Drift")
    axis.set_xlabel("Electronic Timestep [as]")
    axis.set_title("Model %i, Nuclear dt = %.2g as" % (model, dt))


def plotEnerVsNuclDt(allNestedData, model, axis=False):
    """
    Will plot the electronic timestep vs the norm for all the data in the
    nested data input for a certain model. BOB
    """ 
    allData = allNestedData.query_data({'tullyModel': model})
    
    elec_dt = [data.dt / data.elec_steps for data in allData]
    # I would like to put a elec_steps check in here. All elec dt should be the same!
    
    nucl_dt = [data.dt * 24.188843265857 for data in allData]  # in as
    ener_drifts = [abs(data.get_ener_drift()) for data in allData]
    
    if axis is False:
        f, axis = plt.subplots()
    
    axis.plot(nucl_dt, ener_drifts, 'ko', ls='--')
    #axis.set_yscale("log")
    axis.grid(color="#dddddd")
    axis.set_ylabel(r"Ener Drift [$\frac{Ha}{ps \ atom}$]")
    axis.set_xlabel("Nuclear Timestep [as]")
    axis.set_title("Model %i, Elec dt = %.2g as" % (model, elec_dt[0]))

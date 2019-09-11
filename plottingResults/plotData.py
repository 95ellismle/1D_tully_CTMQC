#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:05:11 2019

@author: mellis
"""

import matplotlib.pyplot as plt


def plotNormVsElecDt(allNestedData, model):
    """
    Will plot the electronic timestep vs the norm for all the data in the
    nested data input for a certain model.
    """
    allData = allNestedData.query_data({'tullyModel': model})
    dt = allData[0].dt * 2.4188843265857e1  # in as
    elec_dts = [dt / data.elec_steps for data in allData]
    norms = [abs(data.get_norm_drift()) for data in allData]
    
    f, a = plt.subplots()
    a.plot(elec_dts, norms, 'ko', ls='--')
    a.set_yscale("log")
    a.grid(color="#dddddd")
    a.set_ylabel("Norm Drift")
    a.set_xlabel("Electronic Timestep [as]")
    a.set_title("Model %i, Nuclear dt = %.2g as" % (model, dt))

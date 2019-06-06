# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:13:47 2019

@author: Taylor
"""
import numpy as np
import time

import mutual_information as mi
import mutual_information.data as d
import matplotlib.pyplot as plt

data = d.load_data()

plt.figure(1)
p = mi.pointwise_mutual_information(data['az'],data['sme'], prange = [[-90,90.],[0,1000.]], nbins = 18)
mi.plot_pmi(p, title = '', xlabel = 'Phase Front Angle (theta)', ylabel = 'SME (nT)', cbar = True, cbar_title = 'Information (bits)')


plt.figure(2)
pcmi = mi.pointwise_conditional_mutual_information(data['az'],data['sme'],data['B'][:,2], prange = [[-90,90.],[0,1000.], [-10,10]], nbins = 18, sumoverz = True)
mi.plot_pmi(pcmi, title = '', xlabel = 'Phase Front Azimuth (phi)', ylabel = 'SME (nT)', cbar = True, cbar_title = 'Information (bits)')
plt.plot([45,45],[0,1000],'--', color = 'k')

plt.figure(3)
p = mi.pointwise_mutual_information(data['inc'],data['sme'], prange = [[-90,90.],[0,1000.]], nbins = 18)
mi.plot_pmi(p, title = '', xlabel = 'Phase Front Angle (theta)', ylabel = 'SME (nT)', cbar = True, cbar_title = 'Information (bits)')


plt.figure(4)
pcmi = mi.pointwise_conditional_mutual_information(data['inc'],data['sme'],data['B'][:,2], prange = [[-90,90.],[0,1000.], [-10,10]], nbins = 18, sumoverz = True)
mi.plot_pmi(pcmi, title = '', xlabel = 'Phase Front Azimuth (phi)', ylabel = 'SME (nT)', cbar = True, cbar_title = 'Information (bits)')
plt.plot([45,45],[0,1000],'--', color = 'k')
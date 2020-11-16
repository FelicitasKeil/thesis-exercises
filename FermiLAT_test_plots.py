#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:43:52 2020

@author: felicitaskeil
"""
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from astropy.io import fits
from astropy.table import Table

start = timeit.default_timer() 

f_lat_data = fits.open('gll_psc_v24.fit')

sources = f_lat_data[1].data

sources_cols = f_lat_data[1].columns

bands = np.array([0.05, 0.1, 0.3, 1, 3, 10, 30, 300])

'''filename = 'FLAT_test_pickle.pkl'
with open(filename, 'wb') as dump_pkl:
    pickle.dump(bands, dump_pkl)
    '''
''' -----------------------PLOT ENERGY SPECTRA ----------------------------------------'''
x = np.empty([7])

for i in range(x.size):
    x[i] = np.exp(np.log(bands[i])+(np.log(bands[i+1])-np.log(bands[i]))/2)


fig, axes = plt.subplots(2, 2, squeeze=False)

s1=0
error1=sources.field('Unc_Flux_Band')[s1].transpose()
axes[0,0].errorbar(x, sources.field('Flux_Band')[s1], 
                   yerr=error1, uplims=True, lolims=True, ecolor='#99ccff')
plt.xlabel('Energy in GeV')
plt.ylabel('Photon Flux in cm^(-2) s^(-1)')
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].set_ylim([10**(-15), 10**(-7)])
axes[0,0].set_title('PowerLaw Spectrum')

s2=6
error2=sources.field('Unc_Flux_Band')[s2].transpose()
axes[0,1].errorbar(x, sources.field('Flux_Band')[s2],
                   yerr=error2, c='red', uplims=True, lolims=True, ecolor='#ff9966')
axes[0,1].set_xscale('log')
axes[0,1].set_yscale('log')
axes[0,1].set_ylim([10**(-15), 10**(-7)])
axes[0,1].set_title('LogParabola Spectrum')

s3=7
error3=sources.field('Unc_Flux_Band')[s3].transpose()
axes[1,0].errorbar(x, sources.field('Flux_Band')[s3],
                   yerr=error3, c='green', uplims=True, lolims=True, ecolor='#b3ffb3')
axes[1,0].set_xscale('log')
axes[1,0].set_yscale('log')
axes[1,0].set_ylim([10**(-15), 10**(-7)])
axes[1,0].set_title('PLEC Spectrum')
plt.tight_layout()
plt.savefig('Spectral_Fits_Visualisation', dpi=300)
plt.show()

''' -----------------------VARIABILITY HISTOGRAM --------------------------------------'''

variab_index=sources.field('Variability_Index')
plt.hist(variab_index[variab_index > -1], bins=500)
plt.yscale('log')
plt.xlim(0, 10000)
plt.title('Variability Index Histogram')
plt.savefig('Variability_Visualisation', dpi=300)
plt.show()

''' -----------------------PLOT FLUX HISTORY (10 YEARS) -------------------------------'''
x = np.linspace(1, 10, 10)

fig, axes = plt.subplots(2, 2, squeeze=False)
    
axes[0,0].errorbar(x, sources.field('Flux_History')[169])
plt.xlabel('t in years (Observation 2008-2016')
plt.ylabel('Photon Flux in cm^(-2) s^(-1)')
axes[0,0].set_ylim([3*10**(-9), 8*10**(-8)])
axes[0,0].set_title('Identified BLL')

axes[0,1].plot(x, sources.field('Flux_History')[64], c='red')
axes[0,1].set_ylim([3*10**(-9), 8*10**(-8)])
axes[0,1].set_title('Identified FSRQ')

axes[1,0].plot(x, sources.field('Flux_History')[2], c='green')
axes[1,0].set_ylim([3*10**(-9), 8*10**(-8)])
axes[1,0].set_title('associated bll')

axes[1,1].plot(x, sources.field('Flux_History')[3], c='purple')
axes[1,1].set_ylim([3*10**(-9), 8*10**(-8)])
axes[1,1].set_title('associated fsrq')
plt.tight_layout()
plt.savefig('Flux_History_Visualisation', dpi=300)
plt.show()

f_lat_data.close()

''' -----------------------SPLIT DATA BY LABEL (BLL/FSRQ) ----------------------------'''


dat = Table.read('gll_psc_v24.fit', format='fits')
sources_pd = dat.to_pandas()


'''-------------------------- TIMER ------------------------------------------------'''

stop = timeit.default_timer()                                   #time program
print('\n Execution Time: ', stop - start, 's') 


''' -----------------------RECYCLING BIN----------------------------------------------'''

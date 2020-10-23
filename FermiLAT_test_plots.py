#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:43:52 2020

@author: felicitaskeil
"""
import timeit
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

start = timeit.default_timer() 

f_lat_data = fits.open('gll_psc_v22.fit')

sources = f_lat_data[1].data

sources_cols = f_lat_data[1].columns

bands = np.array([0.05, 0.1, 0.3, 1, 3, 10, 30, 300])

x = np.empty([7])

for i in range(x.size):
    x[i] = np.exp(np.log(bands[i])+(np.log(bands[i+1])-np.log(bands[i]))/2)
    
plt.plot(x, sources.field('nuFnu_Band')[884], '.')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.plot(x, sources.field('nuFnu_Band')[1053], '.', c='red')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.plot(x, sources.field('nuFnu_Band')[7], '.', c='green')
plt.xscale('log')
plt.yscale('log')
plt.show()

f_lat_data.close()


'''-------------------------- TIMER ------------------------------------------------'''

stop = timeit.default_timer()                                   #time program
print('\n Execution Time: ', stop - start, 's') 




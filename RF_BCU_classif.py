#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:01:49 2020

@author: felicitaskeil
"""
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from astropy.io import fits
from catalog import catalog
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import functions

# start program timer
start = timeit.default_timer()
# get Silvia's catalog class
mycatalog = catalog('gll_psc_v24.fit', 1)
mycatalog()

f_lat_data = fits.open('gll_psc_v24.fit')
sources = f_lat_data[1].data
sources_cols = f_lat_data[1].columns

merged = functions.merge_catalogs('gll_psc_v26.fit', 'table_4LAC.fits')
merged = merged[merged.GLON.notnull()]
# create empty label array
label = np.empty(mycatalog.Ns)
class1 = mycatalog.feature('CLASS1').astype(str)
# label BLL as 0, FSRQ as 1
for i in range(label.size):
    if class1[i].strip() == 'bll':
        label[i]=0
    elif class1[i].strip() == 'BLL':
        label[i]=0
    elif class1[i].strip() == 'fsrq':
        label[i]=1
    elif class1[i].strip() == 'FSRQ':
        label[i] = 1
    # label BCUs as 2
    elif class1[i].strip() == 'BCU':
        label[i] = 2
    elif class1[i].strip() == 'bcu':
        label[i] = 2
    else:
        label[i]=np.nan
# DataFrame for histograms
hist_data = {'BZ_Class': label,
             'PL_Index': mycatalog.feature('PL_Index'),
             'Variability_Index': mycatalog.feature('Variability_Index'),
             'Flux1000': mycatalog.feature('Flux1000')}
                              
hist_pd = pd.DataFrame(hist_data, columns = ['BZ_Class', 'PL_Index',
                                             'Variability_Index', 'Flux1000'])
# convert to avoid errors
hist_pd = hist_pd.astype(np.float32)

merged['BZ_Class'] = label.tolist()
# merged['nu_syn'] = merged['nu_syn'].fillna(0)
# merged['nuFnu_syn'] = merged['nuFnu_syn'].fillna(0)

# %% Plot various histograms for data visualisation

hist_pd = merged

bll = hist_pd[hist_pd["BZ_Class"] == 0]
bll_upper = hist_pd[hist_pd["BZ_Class"] == 10]
fsrq = hist_pd[hist_pd["BZ_Class"] == 1]
fsrq_upper = hist_pd[hist_pd["BZ_Class"] == 11]
bcu = hist_pd[hist_pd["BZ_Class"] == 2]

plt.hist(bll["PL_Index"], bins='auto', label='BL Lacertae assoc.',
         color='green', fill=False, histtype='step')
plt.hist(bll_upper["PL_Index"], bins='auto', label='BL Lacertae ident.',
         color='#9DDB8D')
plt.hist(fsrq["PL_Index"], bins='auto', label='FSRQs assoc.',
         color='blue', fill=False, histtype='step')
plt.hist(fsrq_upper["PL_Index"], bins='auto', label='FSRQs ident.',
         color='skyblue')
plt.hist(bcu["PL_Index"], bins='auto', label='Unclassified Blazars',
         color='orange', fill=False, histtype='step')
plt.xlabel('Power Law Photon Index')
plt.ylabel('# of sources')
plt.legend()
plt.savefig('PL_Index_hist.png', dpi=300)
plt.show()


plt.hist(np.log10(bll["Variability_Index"]), bins = 'auto',     #Var Index histogram
         label='BL Lacertae assoc.', color='green', fill=False, histtype='step')
plt.hist(np.log10(bll_upper["Variability_Index"]), bins = 'auto',
         label='BL Lacertae ident.', color='#9DDB8D')
plt.hist(np.log10(fsrq["Variability_Index"]), bins='auto',
         label='FSRQs assoc.', color='blue', fill=False, histtype='step')
plt.hist(np.log10(fsrq_upper["Variability_Index"]), bins='auto',
         label='FSRQs ident.', color='skyblue')
plt.hist(np.log10(bcu["Variability_Index"]), bins='auto', 
         label='Unclassified Blazars', 
         color='orange', fill=False, histtype='step')
plt.xlabel('log10 of Variability Index')
plt.ylabel('# of sources')
plt.legend()
plt.savefig('Var_Index_hist.png', dpi=300)
plt.show()

# Integral Flux histogram
plt.hist(np.log10(bll["Flux1000"]), bins='auto',
         label='BL Lacertae assoc.', color='green',
         fill=False, histtype='step')
# plt.hist(np.log10(bll_upper["Flux1000"]), bins='auto',
#          label='BL Lacertae ident.', color='#9DDB8D')
plt.hist(np.log10(fsrq["Flux1000"]), bins='auto',
         label='FSRQs assoc.', color='blue',
         fill=False, histtype='step')
# plt.hist(np.log10(fsrq_upper["Flux1000"]), bins='auto',
#          label='FSRQs ident.', color='skyblue')
plt.hist(np.log10(bcu["Flux1000"]), bins='auto',
         label='Unclassified Blazars', color='orange',
         fill=False, histtype='step')
plt.yscale('log')
plt.xlabel('log10 of Integral Flux in $cm^{-2}s^{-1}$')
plt.ylabel('# of sources')
plt.legend()
plt.savefig('Flux_1000_hist.png', dpi=300)
plt.show()

# Plot Source Count Distribution

counts, bins = np.histogram(np.log10(bll["Flux1000"]), bins=15)
# bin width
h = bins[1] - bins[0]
# complete solid angle in deg^2
sol_angle_full = 41252.96
bin_middle = np.zeros(len(counts)-1)
bin_middle_long = np.zeros(len(counts))
dN_dS_S2 = np.empty(len(counts)-1)
for i in range(len(counts)-1):
    bin_middle[i] = (bins[i]+bins[i+1])/2
    dN_dS_S2[i] = (counts[i+1]-counts[i])/h*10**(bin_middle[i])/sol_angle_full
    bin_middle_long[i] = bin_middle[i]

bin_middle_long[len(counts)-1] = (bins[len(counts)-1]+bins[len(counts)])/2
# TODO: understand source count distr. better
dN_dS_dO_S2 = np.multiply(counts, 10**bin_middle_long)/sol_angle_full
plt.plot(bin_middle_long, dN_dS_dO_S2, '.')
plt.show()
plt.plot(bin_middle, dN_dS_S2, '.')
plt.show()

'''
bll.drop(axis=0, labels=bll.nu_syn[bll.nu_syn == 0].index, inplace=True)
fsrq.drop(axis=0, labels=fsrq.nu_syn[fsrq.nu_syn == 0].index, inplace=True)
bcu.drop(axis=0, labels=bcu.nu_syn[bcu.nu_syn == 0].index, inplace=True)
bll.drop(axis=0, labels=bll.nu_syn[bll.nu_syn.isna()].index, inplace=True)
fsrq.drop(axis=0, labels=fsrq.nu_syn[fsrq.nu_syn.isna()].index, inplace=True)
bcu.drop(axis=0, labels=bcu.nu_syn[bcu.nu_syn.isna()].index, inplace=True)

plt.hist(bll["nu_syn"], bins='auto',              # Integral Flux histogram
         label='BL Lacertae', color='green')
plt.hist(fsrq["nu_syn"], bins='auto',
         label='FSRQs', color='blue')
plt.hist(bcu["nu_syn"], bins='auto',
         label='Unclassified Blazars', color='orange')
plt.xlabel('Synchrotron Peak in Hz')
plt.ylabel('# of sources')
plt.legend()
# plt.savefig('Flux_1000_hist.png', dpi=300)
plt.show()
'''

'''-------------------------- MULTIDIMENSIONAL COLUMNS --------------------------------'''

Flux_Band1 = np.empty(mycatalog.Ns)                             #create 7 arrays Flux_Band
Flux_Band2 = np.empty(mycatalog.Ns)
Flux_Band3 = np.empty(mycatalog.Ns)
Flux_Band4 = np.empty(mycatalog.Ns)
Flux_Band5 = np.empty(mycatalog.Ns)
Flux_Band6 = np.empty(mycatalog.Ns)
Flux_Band7 = np.empty(mycatalog.Ns)

for i in range(Flux_Band1.size):
    Flux_Band1[i] = sources.field('Flux_Band')[i][0] 
    Flux_Band2[i] = sources.field('Flux_Band')[i][1]
    Flux_Band3[i] = sources.field('Flux_Band')[i][2] 
    Flux_Band4[i] = sources.field('Flux_Band')[i][3]
    Flux_Band5[i] = sources.field('Flux_Band')[i][4] 
    Flux_Band6[i] = sources.field('Flux_Band')[i][5]
    Flux_Band7[i] = sources.field('Flux_Band')[i][6] 


Flux_History1 = np.empty(mycatalog.Ns)                          #create 7 arrays Flux_History
Flux_History2 = np.empty(mycatalog.Ns)
Flux_History3 = np.empty(mycatalog.Ns)
Flux_History4 = np.empty(mycatalog.Ns)
Flux_History5 = np.empty(mycatalog.Ns)
Flux_History6 = np.empty(mycatalog.Ns)
Flux_History7 = np.empty(mycatalog.Ns)
Flux_History8 = np.empty(mycatalog.Ns)
Flux_History9 = np.empty(mycatalog.Ns)
Flux_History10 = np.empty(mycatalog.Ns)

for i in range(Flux_Band1.size):
    Flux_History1[i] = sources.field('Flux_History')[i][0] 
    Flux_History2[i] = sources.field('Flux_History')[i][1]
    Flux_History3[i] = sources.field('Flux_History')[i][2] 
    Flux_History4[i] = sources.field('Flux_History')[i][3]
    Flux_History5[i] = sources.field('Flux_History')[i][4] 
    Flux_History6[i] = sources.field('Flux_History')[i][5]
    Flux_History7[i] = sources.field('Flux_History')[i][6] 
    Flux_History8[i] = sources.field('Flux_History')[i][7] 
    Flux_History9[i] = sources.field('Flux_History')[i][8]
    Flux_History10[i] = sources.field('Flux_History')[i][9] 

'''-------------------------- CREATE DATA FRAME --------------------------------------'''

sources_data = {'BZ_Class': label,                              #create DataFrame w. fluxes
                'Flux_Band1': Flux_Band1,
                'Flux_Band2': Flux_Band2,
                'Flux_Band3': Flux_Band3,
                'Flux_Band4': Flux_Band4,
                'Flux_Band5': Flux_Band5,
                'Flux_Band6': Flux_Band6,
                'Flux_Band7': Flux_Band7,
                'Flux_History1': Flux_History1,
                'Flux_History2': Flux_History2,
                'Flux_History3': Flux_History3,
                'Flux_History4': Flux_History4,
                'Flux_History5': Flux_History5,
                'Flux_History6': Flux_History6,
                'Flux_History7': Flux_History7,
                'Flux_History8': Flux_History8,
                'Flux_History9': Flux_History9,
                'Flux_History10': Flux_History10}

sources_pd = pd.DataFrame(sources_data,
                          columns = ['BZ_Class', 
                                     'Flux_Band1', 'Flux_Band2',
                                     'Flux_Band3', 'Flux_Band4',
                                     'Flux_Band5', 'Flux_Band6',
                                     'Flux_Band7', 'Flux_History1',
                                     'Flux_History2', 'Flux_History3',
                                     'Flux_History4', 'Flux_History5',
                                     'Flux_History6', 'Flux_History7',
                                     'Flux_History8', 'Flux_History9',
                                     'Flux_History10'])

# remove non-blazar sources
sources_pd.drop(axis=0,
                labels=sources_pd.BZ_Class[sources_pd.BZ_Class.isna()].index,
                inplace=True)
# created DataFrame without BCUs
sources_assoc = sources_pd[sources_pd.BZ_Class != 2]

bz_amount = np.shape(sources_assoc)[0]
print('There are', bz_amount, 'associated blazar sources:'
      '{0:.0f} in the training set & {1:.0f} in the test set.'.format(0.8*bz_amount, 0.2*bz_amount))

'''-------------------------- RANDOM FOREST CLASSIFIER --------------------------------'''

split = StratifiedShuffleSplit(n_splits=1,                      #train/test split stratified
                               test_size=0.2, random_state=72)
for train_index, test_index in split.split(sources_assoc, sources_assoc["BZ_Class"]):
    train_set = sources_assoc.iloc[train_index]
    test_set = sources_assoc.iloc[test_index]
    
train_X = train_set.drop(["BZ_Class"], axis=1)                  #separate label column (Y)
train_Y = train_set["BZ_Class"]
test_X = test_set.drop(["BZ_Class"], axis=1)
test_Y = test_set["BZ_Class"]

train_X = train_X.astype(np.float32)                            #convert to avoid NaN later
train_Y = train_Y.astype(np.float32)
train_X = np.nan_to_num(train_X)                                # set NaN=0
train_Y = np.nan_to_num(train_Y)

scaler = StandardScaler()

train_X_transf = scaler.fit_transform(train_X, train_Y)

# apply RF to training set
forest_clf = RandomForestClassifier()
forest_clf.fit(train_X_transf, train_Y)

# predict prob. w. CV
y_probas_forest = cross_val_predict(forest_clf,
                                    train_X_transf, train_Y, cv=10,
                                    method="predict_proba")

train_rmse = np.sqrt(mean_squared_error(train_Y, y_probas_forest[:, 1]))
print('\n Root Mean Squared Error for the train set:', train_rmse)

predictions = np.zeros(train_Y.size)                            # make classifications
correct = 0                                                     # count the correct ones
for i in range(train_Y.size):
    if y_probas_forest[i, 1] >= 0.5:
        predictions[i] = 1
    if predictions[i] == train_Y[i]:
        correct += 1

print('\n Amount of correctly classified training sources:', correct)

# percentage w. correct class
accuracy = correct/train_Y.size
print('\n Training Accuracy:', accuracy)


# %% TESTING

# convert to avoid NaN later
test_X = test_X.astype(np.float32)
test_X = np.nan_to_num(test_X)
test_Y = np.nan_to_num(test_Y)

test_X_transf = scaler.transform(test_X)                        # scale & predict 
test_probas = forest_clf.predict_proba(test_X_transf)

test_rmse = np.sqrt(mean_squared_error(test_Y, test_probas[:, 1]))
print('\n Root Mean Squared Error for the test set:', test_rmse)

test_predictions = np.zeros(test_Y.size)                        # make & check classific.
correct_test = 0
for i in range(test_Y.size):
    if test_probas[i, 1] >= 0.5:
        test_predictions[i] = 1
    if test_predictions[i] == test_Y[i]:
        correct_test += 1

print('\n Amount of correctly classified test sources:', correct_test)

accuracy = correct_test/test_Y.size                             # test accuracy
print('\n Testing Accuracy:', accuracy)

# %% BCU CLASSIFICATION

# take sources with class 2 (see above)
sources_bcu = sources_pd[sources_pd.BZ_Class == 2]

# drop column with the 2s
bcu_X = sources_bcu.drop(["BZ_Class"], axis=1)
bcu_X = bcu_X.astype(np.float32)
bcu_X = np.nan_to_num(bcu_X)

bcu_X_transf = scaler.transform(bcu_X)
# predict class probabilities for the BCUs
bcu_probas = forest_clf.predict_proba(bcu_X_transf)

bcu_amount = np.shape(sources_bcu)[0]

# make classifications from probabilities
bcu_predictions = np.zeros(bcu_amount)
for i in range(bcu_amount):
    if bcu_probas[i, 1] >= 0.5:
        bcu_predictions[i] = 1

print('Those are the predictions for all BCUs; 0 = BLL; 1 = FSRQ',
      bcu_predictions)

predictions_assoc = np.array(['fsrq' if x == 1 else 'bll'
                              for x in bcu_predictions])
bcu['BZ_Class'] = predictions_assoc

fsrq_list = []
bll_list = []
bcu_index = bcu.reset_index()

for i in range(len(bcu_index)):
    if bcu_index.iloc[i]['BZ_Class'] == 'fsrq':
        fsrq_list.append([bcu_index.iloc[i]['Source_Name'],
                          bcu_index.iloc[i]['ASSOC1']])
    elif bcu_index.iloc[i]['BZ_Class'] == 'bll':
        bll_list.append([bcu_index.iloc[i]['Source_Name'],
                         bcu_index.iloc[i]['ASSOC1']])

# %% TIMER

stop = timeit.default_timer()
print('\n Execution Time: ', stop - start, 's')


# %% RECYCLING BIN

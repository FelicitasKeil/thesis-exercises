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


start = timeit.default_timer()                                  #start program timer

mycatalog = catalog('gll_psc_v24.fit', 1)                       #get Silvia's catalog class
mycatalog()

f_lat_data = fits.open('gll_psc_v24.fit')
sources = f_lat_data[1].data
sources_cols = f_lat_data[1].columns

label=np.empty(mycatalog.Ns)                                    #create empty label array
class1 = mycatalog.feature('CLASS1').astype(str)

for i in range(label.size):                                     #label BLL as 0, FSRQ as 1
    if class1[i].strip() == 'bll':
        label[i]=0
    elif class1[i].strip() == 'BLL':
        label[i]=0
    elif class1[i].strip() == 'fsrq':
        label[i]=1
    elif class1[i].strip() == 'FSRQ':
        label[i]=1
    elif class1[i].strip() == 'BCU':                            #label BCUs as 2
        label[i]=2
    elif class1[i].strip() == 'bcu':
        label[i]=2
    else:
        label[i]=np.nan
        
hist_data = {'BZ_Class': label,                                 #DataFrame for histograms
             'PL_Index': mycatalog.feature('PL_Index'),
             'Variability_Index': mycatalog.feature('Variability_Index'),
             'Flux1000': mycatalog.feature('Flux1000')}
                              
hist_pd = pd.DataFrame(hist_data, columns = ['BZ_Class', 'PL_Index',
                                             'Variability_Index', 'Flux1000'])

hist_pd = hist_pd.astype(np.float32)                            #convert to avoid errors

bll = hist_pd[hist_pd["BZ_Class"]==0]
fsrq = hist_pd[hist_pd["BZ_Class"]==1]
bcu = hist_pd[hist_pd["BZ_Class"]==2]


plt.hist(bll["PL_Index"],bins = 'auto', label='BL Lacertae',    #PL Index histogram
         color='green', fill=False, histtype='step') 
plt.hist(fsrq["PL_Index"], bins='auto', label='FSRQs', 
         color='blue', fill=False, histtype='step')
plt.hist(bcu["PL_Index"], bins='auto', label='Unclassified Blazars',
         color='orange', fill=False, histtype='step')
plt.xlabel('Power Law Photon Index')
plt.ylabel('# of sources')
plt.legend()
plt.savefig('PL_Index_hist.png', dpi=300)
plt.show()


plt.hist(np.log10(bll["Variability_Index"]), bins = 'auto',     #Var Index histogram
         label='BL Lacertae', color='green', fill=False, histtype='step')
plt.hist(np.log10(fsrq["Variability_Index"]), bins='auto', 
         label='FSRQs', color='blue', fill=False, histtype='step')
plt.hist(np.log10(bcu["Variability_Index"]), bins='auto', 
         label='Unclassified Blazars', 
         color='orange', fill=False, histtype='step')
plt.xlabel('log10 of Variability Index')
plt.ylabel('# of sources')
plt.legend()
plt.savefig('Var_Index_hist.png', dpi=300)
plt.show()


plt.hist(np.log10(bll["Flux1000"]), bins = 'auto',              #Integral Flux histogram
         label='BL Lacertae', color='green', 
         fill=False, histtype='step') 
plt.hist(np.log10(fsrq["Flux1000"]), bins='auto', 
         label='FSRQs', color='blue', 
         fill=False, histtype='step')
plt.hist(np.log10(bcu["Flux1000"]), bins='auto', 
         label='Unclassified Blazars', color='orange', 
         fill=False, histtype='step')
plt.xlabel('log10 of Integral Flux in $cm^{-2}s^{-1}$')
plt.ylabel('# of sources')
plt.legend()
plt.savefig('Flux_1000_hist.png', dpi=300)
plt.show()

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
    Flux_History5[i] = sources.field('Flux_History')[i][7] 
    Flux_History6[i] = sources.field('Flux_History')[i][8]
    Flux_History7[i] = sources.field('Flux_History')[i][9] 

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

sources_pd = pd.DataFrame(sources_data, columns = ['BZ_Class', 
                                                   'Flux_Band1', 'Flux_Band2',
                                                   'Flux_Band3', 'Flux_Band4',
                                                   'Flux_Band5', 'Flux_Band6',
                                                   'Flux_Band7', 'Flux_History1', 
                                                   'Flux_History2', 'Flux_History3',
                                                   'Flux_History4', 'Flux_History5',
                                                   'Flux_History6', 'Flux_History7',
                                                   'Flux_History8', 'Flux_History9',
                                                   'Flux_History10'])

sources_pd.drop(axis=0,                                         #remove non-blazar sources 
                labels=sources_pd.BZ_Class[sources_pd.BZ_Class.isna()].index, 
                inplace=True) 

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
train_X = np.nan_to_num(train_X)                                #set NaN=0
train_Y = np.nan_to_num(train_Y)

scaler = RobustScaler()                                         #robust scaler for accuracy

train_X_transf = scaler.fit_transform(train_X, train_Y)         #scale data (Gaussian like)

forest_clf = RandomForestClassifier()                           #apply RF to training set
forest_clf.fit(train_X_transf, train_Y)

y_probas_forest = cross_val_predict(forest_clf,                 #predict prob. w. CV
                                    train_X_transf, train_Y, cv=10,  
                                    method = "predict_proba")

train_rmse = np.sqrt(mean_squared_error(train_Y, y_probas_forest[:,1]))
print('\n Root Mean Squared Error for the train set:', train_rmse)

predictions = np.zeros(train_Y.size)                            #make classifications
correct = 0                                                     #count the correct ones
for i in range(train_Y.size):
    if y_probas_forest[i, 1] >= 0.5:
        predictions[i] = 1
    if predictions[i] == train_Y[i]:
        correct += 1

print('\n Amount of correctly classified training sources:', correct)

accuracy = correct/train_Y.size                                 #percentage w. correct class
print('\n Training Accuracy:', accuracy)



'''-------------------------- TESTING -------------------------------------------------'''

test_X = test_X.astype(np.float32)                              #convert to avoid NaN later
test_X = np.nan_to_num(test_X)
test_Y = np.nan_to_num(test_Y)

test_X_transf = scaler.transform(test_X)                        #scale & predict 
test_probas = forest_clf.predict_proba(test_X_transf)

test_rmse = np.sqrt(mean_squared_error(test_Y, test_probas[:,1]))
print('\n Root Mean Squared Error for the test set:', test_rmse)

test_predictions = np.zeros(test_Y.size)                        #make & check classific.
correct_test = 0
for i in range(test_Y.size):
    if test_probas[i, 1] >= 0.5:
        test_predictions[i] = 1
    if test_predictions[i] == test_Y[i]:
        correct_test += 1

print('\n Amount of correctly classified test sources:', correct_test)

accuracy = correct_test/test_Y.size                             #test accuracy
print('\n Testing Accuracy:', accuracy)

'''-------------------------- BCU CLASSIFICATION --------------------------------------'''

sources_bcu = sources_pd[sources_pd.BZ_Class == 2]

bcu_X = sources_bcu.drop(["BZ_Class"], axis = 1)
bcu_X = bcu_X.astype(np.float32)
bcu_X = np.nan_to_num(bcu_X)

bcu_X_transf = scaler.transform(bcu_X)                          #scale & predict 
bcu_probas = forest_clf.predict_proba(bcu_X_transf)

bcu_amount = np.shape(sources_bcu)[0]

bcu_predictions = np.zeros(bcu_amount)                          #make classifications
for i in range(bcu_amount):
    if bcu_probas[i, 1] >= 0.5:
        bcu_predictions[i] = 1

print('Those are the predictions for all BCUs; 0 = BLL; 1 = FSRQ', bcu_predictions)

'''-------------------------- TIMER ---------------------------------------------------'''

stop = timeit.default_timer()                                   #time program
print('\n Execution Time: ', stop - start, 's') 


''' -----------------------RECYCLING BIN----------------------------------------------'''

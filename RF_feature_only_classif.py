#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:19:03 2020

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
from sklearn import tree


start = timeit.default_timer()                                  #start program timer

'''-------------------------- LABELLING --------------------------------------------'''

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
        
sources_pd = mycatalog.pdTable                                  #convert to pd DataFrame

sources_pd['BZ_Class'] = label                                  #append created label

'''-------------------------- DATA PREPARATION ---------------------------------------'''

sources_pd.drop(axis=0,                                         #remove non-blazar sources 
                labels=sources_pd.BZ_Class[sources_pd.BZ_Class.isna()].index, 
                inplace=True) 
       
sources_pd.replace(r'^\s*$', np.nan, regex=True, inplace=True)

for feature in sources_pd.columns:
    if sources_pd[feature].isna().sum() > 0.05 * len(sources_pd[feature]):
        sources_pd.drop(feature, axis=1, inplace=True)

drop_cols = [col for col in sources_pd if col.startswith('ASSOC')]
drop_cols = np.append (drop_cols, 
                       [col for col in sources_pd if col.startswith('Conf')])
drop_cols = np.append (drop_cols, 
                       [col for col in sources_pd if col.startswith('Unc')])
drop_cols = np.append (drop_cols, 
                       [col for col in sources_pd if col.endswith('Peak')])
drop_cols = np.append(drop_cols,
                      ['Source_Name', 'SpectrumType', 'RA_Counterpart', 
                       'DEC_Counterpart', 'Flags', 'TEVCAT_FLAG', 'CLASS1',
                       'Peak_Interval', 'ROI_num', 'Npred', 'DataRelease',
                       'RAJ2000', 'DEJ2000','GLON', 'GLAT'])

sources_pd = sources_pd.drop(drop_cols, axis=1)

bz_amount = np.shape(sources_pd)[0]
print('There are', bz_amount, 'sources; '
      '{0:.0f} in the training set & {1:.0f} in the test set.'.format(0.8*bz_amount, 0.2*bz_amount))


'''-------------------------- SCATTER PLOTS -------------------------------------------'''

bll = sources_pd[sources_pd["BZ_Class"]==0]
fsrq = sources_pd[sources_pd["BZ_Class"]==1]
bcu = sources_pd[sources_pd["BZ_Class"]==2]


'''-------------------------- TRAINING -------------------------------------------------'''

sources_assoc = sources_pd[sources_pd.BZ_Class != 2]            #keep associated sources

split = StratifiedShuffleSplit(n_splits=10,                      #train/test split stratified
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

scaler = StandardScaler()                                       

train_X_transf = scaler.fit_transform(train_X, train_Y)         #scale data (Gaussian like)

#print('\n train mean, variance: ', 
#      np.mean(train_X_transf), np.var(train_X_transf))

forest_clf = RandomForestClassifier(n_estimators=100,           #apply RF to training set
                                    max_depth=None, criterion="entropy")                           
forest_clf.fit(train_X_transf, train_Y)

y_probas_forest = cross_val_predict(forest_clf,                 #predict prob. w. CV
                                    train_X_transf, train_Y, 
                                    cv=10, method = "predict_proba")

forest_params = forest_clf.get_params()
print('\n RF parameters: ', forest_params)

feature_imp = forest_clf.feature_importances_
print('\n Feature importances: ', feature_imp)

#indicator, n_nodes_ptr = forest_clf.decision_path(train_X_transf)

tree_depth = np.empty(forest_clf.n_estimators)
for i in range(forest_clf.n_estimators):
    tree_depth[i] = forest_clf.estimators_[i].tree_.max_depth
    
print('Mean tree depth', np.mean(tree_depth))

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
#print('\n test mean, variance: ', 
#      np.mean(test_X_transf), np.var(test_X_transf))

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

'''-------------------------- TIMER ---------------------------------------------------'''

stop = timeit.default_timer()                                   #time program
print('\n Execution Time: ', stop - start, 's') 


''' -----------------------RECYCLING BIN----------------------------------------------'''

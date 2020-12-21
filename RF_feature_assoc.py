
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:47:10 2020

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

'''-------------------------- LABELLING --------------------------------------------'''
#%%
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
        label[i]=10
    elif class1[i].strip() == 'fsrq':
        label[i]=1
    elif class1[i].strip() == 'FSRQ':
        label[i]=11
    elif class1[i].strip() == 'BCU':                            #label BCUs as 2
        label[i]=2
    elif class1[i].strip() == 'bcu':
        label[i]=2
    else:
        label[i]=np.nan
        
sources_pd = mycatalog.pdTable                                  #convert to pd DataFrame

sources_pd['BZ_Class'] = label                                  #append created label

'''-------------------------- MULTIDIMENSIONAL COLUMNS --------------------------------'''
#%%
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
#%%
multidim_data = {'Flux_Band1': Flux_Band1,                       #create DataFrame w. fluxes
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

multidim_pd = pd.DataFrame(multidim_data, columns = ['Flux_Band1', 'Flux_Band2',
                                                     'Flux_Band3', 'Flux_Band4',
                                                     'Flux_Band5', 'Flux_Band6',
                                                     'Flux_Band7', 'Flux_History1', 
                                                     'Flux_History2', 'Flux_History3',
                                                     'Flux_History4', 'Flux_History5',
                                                     'Flux_History6', 'Flux_History7',
                                                     'Flux_History8', 'Flux_History9',
                                                     'Flux_History10'])
#result = pd.concat([df1, df4], axis=1, sort=False)

sources_concat=pd.concat([sources_pd, multidim_pd], axis=1, sort=False)
sources_pd=sources_concat

'''-------------------------- DATA PREPARATION ---------------------------------------'''
#%%
sources_pd.drop(axis=0,                                         #remove non-blazar sources 
                labels=sources_pd.BZ_Class[sources_pd.BZ_Class.isna()].index, 
                inplace=True) 
       
sources_pd.replace(r'^\s*$', np.nan, regex=True, inplace=True)

for feature in sources_pd.columns:
    if sources_pd[feature].isna().sum() > 0.05 * len(sources_pd[feature]):
        sources_pd.drop(feature, axis=1, inplace=True)

drop_cols = [col for col in sources_pd if col.startswith('ASSOC')]
#drop_cols = np.append (drop_cols, 
#                       [col for col in sources_pd if col.startswith('Conf')])
drop_cols = np.append(drop_cols,
                      ['Source_Name', 'SpectrumType', 'RA_Counterpart', 
                       'DEC_Counterpart', 'Flags', 'TEVCAT_FLAG', 'CLASS1',
                       ])

sources_pd = sources_pd.drop(drop_cols, axis=1)
sources_assoc = sources_pd[sources_pd.BZ_Class != 2]            #keep associated sources

bz_amount = np.shape(sources_assoc)[0]
bll_amount = np.sum(label == 0)
fsrq_amount = np.sum(label == 1)
bcu_amount = np.sum(label == 2)
print('There are', bz_amount, 'sources; '
      '{0:.0f} in the training set & {1:.0f} in the test set.'.format(0.8*bz_amount, 0.2*bz_amount))
print('\nThere are', bll_amount, 'BL Lacs, ', fsrq_amount, 'FSRQs and',
      bcu_amount, 'BCUs.')

# %% SCATTER PLOTS
bll = sources_pd[sources_pd["BZ_Class"] == 0]
bll_upper = sources_pd[sources_pd["BZ_Class"] == 10]
fsrq = sources_pd[sources_pd["BZ_Class"] == 1]
fsrq_upper = sources_pd[sources_pd["BZ_Class"] == 11]
bcu = sources_pd[sources_pd["BZ_Class"] == 2]

# Scatter PL_Ind vs. Variab
plt.scatter(np.log10(bll["PL_Flux_Density"]),
            np.log10(bll["Variability_Index"]),
            label='BL Lacs assoc.', color='green', edgecolors='none',
            alpha=0.5, marker='.')
plt.scatter(np.log10(bll_upper["PL_Flux_Density"]), 
            np.log10(bll_upper["Variability_Index"]),
            label='BL Lacs ident.', color='blue', edgecolors='none',
            marker='.')
plt.scatter(np.log10(fsrq["PL_Flux_Density"]),
            np.log10(fsrq["Variability_Index"]), label='FSRQs assoc.',
            color='red', edgecolors='none', alpha=0.5, marker='.')
plt.scatter(np.log10(fsrq_upper["PL_Flux_Density"]),
            np.log10(fsrq_upper["Variability_Index"]), label='FSRQs ident.',
            color='purple', edgecolors='none', marker='.')
plt.scatter(np.log10(bcu["PL_Flux_Density"]),
            np.log10(bcu["Variability_Index"]), label='BCUs',
            color='orange', edgecolors='none', alpha=0.35, marker='.')
plt.xlabel('$log_{10}$ Differential Flux at Pivot Energy in $cm^{-2}MeV^{-1}s^{-1}$')
plt.ylabel('$log_{10}$ of Variability Index')
plt.legend()
plt.savefig('Flux_Density_Variab_scatter.png', dpi=300)
plt.show()

# Pivot_E vs. Signif
plt.scatter(np.log10(bll["Pivot_Energy"]),np.log10(bll["Signif_Avg"]),
            label='BL Lacs', color='green', edgecolors='none', 
            alpha=0.5, marker='.')
plt.scatter(np.log10(bll_upper["Pivot_Energy"]),
            np.log10(bll_upper["Signif_Avg"]),
            label='BL Lacs', color='blue', edgecolors='none',
            marker='.')
plt.scatter(np.log10(fsrq["Pivot_Energy"]), np.log10(fsrq["Signif_Avg"]), 
            label='FSRQs', color='red', edgecolors='none',
            alpha=0.5, marker='.')
plt.scatter(np.log10(fsrq_upper["Pivot_Energy"]),
            np.log10(fsrq_upper["Signif_Avg"]),
            label='FSRQs', color='purple', edgecolors='none',
            marker='.') 
plt.scatter(np.log10(bcu["Pivot_Energy"]), np.log10(bcu["Signif_Avg"]),
            label='BCUs', color='orange', edgecolors='none',
            alpha=0.35, marker='.')
plt.xlabel('log10 of Pivot Energy in MeV')
plt.ylabel('log10 of Source Significance in $\sigma$')
plt.legend()
plt.savefig('Pivot_E_Signif_scatter.png', dpi=300)
plt.show()

# int flux vs. latitude
plt.scatter(np.log10(bll["Flux1000"]), bll["PL_Index"],
            label='BL Lacs', color='green', edgecolors='none',
            alpha=0.5, marker='.')
plt.scatter(np.log10(bll_upper["Flux1000"]), bll_upper["PL_Index"],
            label='BL Lacs', color='blue', edgecolors='none',
            marker='.')
plt.scatter(np.log10(fsrq["Flux1000"]), fsrq["PL_Index"],
            label='FSRQs', color='red', edgecolors='none',
            alpha=0.5, marker='.')
plt.scatter(np.log10(fsrq_upper["Flux1000"]), fsrq_upper["PL_Index"],
            label='FSRQs', color='purple', edgecolors='none',
            marker='.')
plt.scatter(np.log10(bcu["Flux1000"]), bcu["PL_Index"],
            label='BCUs', color='orange', edgecolors='none',
            alpha=0.35, marker='.')
plt.xlabel('$log_{10}$ of Integral photon flux in $cm^{-2}s^{-1}$')
plt.ylabel('Power Law Photon Index')
plt.legend()
plt.savefig('Flux_PL_Index_scatter.png', dpi=300)
plt.show()


'''-------------------------- TRAINING -------------------------------------------------'''
#%%

split = StratifiedShuffleSplit(n_splits=10,                     #train/test split stratified
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

forest_clf = RandomForestClassifier(criterion="entropy")        #apply RF to training set
forest_clf.fit(train_X_transf, train_Y)

y_probas_forest = cross_val_predict(forest_clf,                 #predict prob. w. CV
                                    train_X_transf, train_Y, 
                                    cv=10, method = "predict_proba")

forest_params = forest_clf.get_params()                         #get forest parameters
print('RF parameters: ', forest_params)

feature_imp = forest_clf.feature_importances_
print('Feature importances: ', feature_imp)

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
#%%
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

'''-------------------------- TIMER ---------------------------------------------------'''
#%%
stop = timeit.default_timer()                                   #time program
print('\n Execution Time: ', stop - start, 's') 


''' -----------------------RECYCLING BIN----------------------------------------------'''

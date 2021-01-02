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
import tensorflow.keras as K
from astropy.io import fits
from catalog import catalog
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance_matrix
import sys
import functions

# start program timer
start = timeit.default_timer()

# %% LABELLING

# get Silvia's catalog class
mycatalog = catalog('gll_psc_v26.fit', 1)
mycatalog()

f_lat_data = fits.open('gll_psc_v26.fit')
sources = f_lat_data[1].data
sources_cols = f_lat_data[1].columns

# create empty label array
label = np.empty(mycatalog.Ns)
class1 = mycatalog.feature('CLASS1').astype(str)

# label BLL as 0, FSRQ as 1
for i in range(label.size):
    if class1[i].strip() == 'bll':
        label[i] = 0
    elif class1[i].strip() == 'BLL':
        label[i] = 0
    elif class1[i].strip() == 'fsrq':
        label[i] = 1
    elif class1[i].strip() == 'FSRQ':
        label[i] = 1
    elif class1[i].strip() == 'BCU':
        label[i] = 2
    elif class1[i].strip() == 'bcu':
        label[i] = 2
    else:
        label[i] = np.nan

sources_pd = mycatalog.pdTable

# merge with 4LAC
merged = functions.merge_catalogs('gll_psc_v26.fit', 'table_4LAC.fits')
merged = merged[merged.GLON.notnull()]
# sources_pd = merged

# append created label
sources_pd['BZ_Class'] = label
bcu = sources_pd[sources_pd["BZ_Class"] == 2]

# %% MULTIDIMENSIONAL COLUMNS

# create 7 arrays Flux_Band
Flux_Band1 = np.empty(mycatalog.Ns)
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

# create 7 arrays Flux_History
Flux_History1 = np.empty(mycatalog.Ns)
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

# %% CREATE DATA FRAME

# create DataFrame w. fluxes
multidim_data = {'Flux_Band1': Flux_Band1,
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

multidim_pd = pd.DataFrame(multidim_data,
                           columns=['Flux_Band1', 'Flux_Band2',
                                    'Flux_Band3', 'Flux_Band4',
                                    'Flux_Band5', 'Flux_Band6',
                                    'Flux_Band7', 'Flux_History1',
                                    'Flux_History2', 'Flux_History3',
                                    'Flux_History4', 'Flux_History5',
                                    'Flux_History6', 'Flux_History7',
                                    'Flux_History8', 'Flux_History9',
                                    'Flux_History10'])
# result = pd.concat([df1, df4], axis=1, sort=False)

sources_concat = pd.concat([sources_pd, multidim_pd],
                           axis=1, sort=False)
sources_pd = sources_concat

# %% DATA PREPARATION

# remove non-blazar sources
sources_pd.drop(axis=0,
                labels=sources_pd.BZ_Class[sources_pd.BZ_Class.isna()].index,
                inplace=True)

sources_pd.replace(r'^\s*$', np.nan, regex=True, inplace=True)

'''
median_nu_syn = np.nanmedian(sources_pd['nu_syn'])
# replace NaN and 0 with median -> change code if 0 is physical
sources_pd['nu_syn'] = sources_pd['nu_syn'].replace(0, np.nan)
sources_pd['nu_syn'] = sources_pd['nu_syn'].fillna(median_nu_syn)

median_nuFnu_syn = np.nanmedian(sources_pd['nuFnu_syn'])
# replace NaN and 0 with median -> change code if 0 is physical
sources_pd['nuFnu_syn'] = sources_pd['nuFnu_syn'].replace(0, np.nan)
sources_pd['nuFnu_syn'] = sources_pd['nuFnu_syn'].fillna(median_nuFnu_syn)
'''

# drop features with more than 10% missing entries
for feature in sources_pd.columns:
    if sources_pd[feature].isna().sum() > 0.1 * len(sources_pd[feature]):
        sources_pd.drop(feature, axis=1, inplace=True)

# drop irrelevatn features
drop_cols = [col for col in sources_pd if col.startswith('ASSOC')]
drop_cols = np.append(drop_cols,
                      ['Source_Name', 'SpectrumType', 'RA_Counterpart',
                       'DEC_Counterpart', 'Flags', 'TEVCAT_FLAG', 'CLASS1'])

sources_pd = sources_pd.drop(drop_cols, axis=1)
# keep associated sources
sources_assoc = sources_pd[sources_pd.BZ_Class != 2]

bz_amount = np.shape(sources_assoc)[0]
bll_amount = np.sum(label == 0)
fsrq_amount = np.sum(label == 1)
bcu_amount = np.sum(label == 2)

print('There are', bz_amount, 'sources; '
      '{0:.0f} in the training set & {1:.0f} in the test set.'.format(0.8*bz_amount, 0.2*bz_amount))
print('\nThere are', bll_amount, 'BL Lacs, ', fsrq_amount, 'FSRQs and', bcu_amount, 'BCUs.')


# %% AUGMENTATION


def data_augmentation(data, labels, nPerClass=None, style='repeat'):
    """Perform data augmentation either to balance two classes or to obtain an
    incresead number of datapoints in both classes.

    Args:
        data (tuple): Tuple of training data
        labels (ndarray): Labels corresponding to data
        nPerClass (int, optional): If a number is given,
            perform data augmentation
            on both classes to obtain this number of sources. Defaults to None.
        style (str, optional): The augmentation style to be used.
            Defaults to 'repeat'.
    Returns:
        Tuple: (augmented, labels) containing augmented data and corresponding
            labels
    """

    def repeat_aug(class_data, nAug):
        if nAug <= 0:
            return None
        rand_inds = np.random.choice(len(class_data[0]), int(nAug))
        augmented = class_data[rand_inds]
        # augmented = tuple(x[rand_inds] for x in class_data)
        return augmented

    def smote_aug(class_data, nAug):
        def calc_distances():
            """Calculate the feature wise distance for each pair of class_data
            distributions using the Wasserstein metric.

            Returns:
                ndarray: (nFeatures, nSamples, nSamples) Array w.
                        pairwise distances
                        for each feature
            """
            # Distance matrix with shape (nFeatures, nSamples, nSamples)
            distances = np.zeros((class_data.shape[0],
                                  class_data.shape[0]))
            # for data in class_data:
            #     data = np.reshape(data, (data.shape[0], -1))
            distances = distance_matrix(class_data, class_data)

            return distances
        if nAug <= 0:
            return None
        dists = calc_distances()

        # cl_data_ls = list(class_data)
        # for ind1, data in enumerate(class_data):
        indices = np.zeros((nAug, 2)).astype(int)
        for ind in range(nAug):
            neighbours = dists[ind]
            k_nn = np.argsort(neighbours)
            top_k_nn = k_nn[1:10]
            # choose a near neighbour out of the top 10
            nn_ind = np.random.choice(top_k_nn, 1)
            indices[ind, 0] = ind
            indices[ind, 1] = nn_ind

        # augmented = tuple(np.array([x[i] + np.random.uniform() * (x[j] - x[i])
        #                             for i, j in indices]) for x in class_data)
        augmented = np.zeros((nAug, class_data.shape[1]))
        for ind in range(nAug):
            i = indices[ind, 0]
            j = indices[ind, 1]
            augmented[i, :] = np.array(class_data[i, :] + np.random.uniform() *
                                       (class_data[j] - class_data[i]))

        return augmented

    print(f'\nUsing augmented data, style: {style}\n')
    if labels.shape[-1] == 2:
        labels = labels.argmax(-1)

    nPos = labels.sum()
    nNeg = len(labels) - nPos
    if nPerClass is None:
        nPerClass = np.max([nPos, nNeg])

    pos_data = data[labels == 0]
    neg_data = data[labels == 1]
    # pos_data = tuple(x[labels == 1] for x in data)
    # neg_data = tuple(x[labels == 0] for x in data)

    if style == 'repeat':
        pos_aug = repeat_aug(pos_data, int(nPerClass - nPos))
        neg_aug = repeat_aug(neg_data, int(nPerClass - nNeg))
    elif style == 'smote':
        pos_aug = smote_aug(pos_data, int(nPerClass - nPos))
        neg_aug = smote_aug(neg_data, int(nPerClass - nNeg))
    else:
        sys.exit("No supported augmentation style")

    if pos_aug is None:
        augmented = neg_aug
        augmented_y = np.zeros(int(nPerClass - nNeg))
    elif neg_aug is None:
        augmented = pos_aug
        augmented_y = np.ones(int(nPerClass - nPos))
    else:
        print('Warning: augmenting both classes')
        # augmented = tuple(np.append(x, y, axis=0)
        #                   for x, y in zip(pos_aug, neg_aug))
        # augmented_y = np.append(
        #     np.ones(len(pos_aug[0])), np.zeros(len(neg_aug[0])))

    return augmented, augmented_y


# %% TRAINING


def forest_train(sources, nTrain, augmentation='smote'):
    print('\nStarting Training', nTrain+1)
    split = StratifiedShuffleSplit(n_splits=10,
                                   test_size=0.2, random_state=72)
    for train_index, test_index in split.split(sources_assoc, sources_assoc["BZ_Class"]):
        train_set = sources_assoc.iloc[train_index]
        test_set = sources_assoc.iloc[test_index]

    # separate label column (Y)
    train_X = train_set.drop(["BZ_Class"], axis=1)
    train_Y = train_set["BZ_Class"]
    test_X = test_set.drop(["BZ_Class"], axis=1)
    test_Y = test_set["BZ_Class"]

    # convert to avoid NaN later, set NaN to 0
    train_X = train_X.astype(np.float32)
    train_Y = train_Y.astype(np.float32)
    train_X = np.nan_to_num(train_X)
    train_Y = np.nan_to_num(train_Y)
    
    if augmentation is not None:
        train_X_aug, train_Y_aug = data_augmentation(train_X, train_Y, style=augmentation)
        concat_X = np.concatenate((train_X, train_X_aug), axis=0)
        train_X = concat_X
        concat_Y = np.concatenate((train_Y, train_Y_aug), axis=0)
        train_Y = concat_Y
        # shuffle rows
        indices = np.random.permutation(train_X.shape[0])
        train_X = train_X[indices]
        train_Y = train_Y[indices]
   
    scaler = StandardScaler()
    train_X_transf = scaler.fit_transform(train_X, train_Y)

    forest_clf = RandomForestClassifier(criterion="entropy")
    forest_clf.fit(train_X_transf, train_Y)

    # 10 fold cross validation
    y_probas_forest = cross_val_predict(forest_clf,
                                        train_X_transf, train_Y,
                                        cv=10, method="predict_proba")

    forest_params = forest_clf.get_params()
    print('RF parameters: ', forest_params)

    feature_imp = forest_clf.feature_importances_
    print('Feature importances: ', feature_imp)

    tree_depth = np.empty(forest_clf.n_estimators)
    for i in range(forest_clf.n_estimators):
        tree_depth[i] = forest_clf.estimators_[i].tree_.max_depth

    print('Mean tree depth', np.mean(tree_depth))

    train_rmse = np.sqrt(mean_squared_error(train_Y, y_probas_forest[:, 1]))
    print('\n Root Mean Squared Error for the train set:', train_rmse)

    # make classifications
    predictions = np.zeros(train_Y.size)
    # count the correct ones
    correct_train = 0
    for i in range(train_Y.size):
        if y_probas_forest[i, 1] >= 0.5:
            predictions[i] = 1
        if predictions[i] == train_Y[i]:
            correct_train += 1

    print('\n Amount of correctly classified training sources:', correct_train)

    # percentage w. correct class
    accuracy_train = correct_train/train_Y.size
    print('\n Training Accuracy:', accuracy_train)

    # TESTING
    
    # convert to avoid NaN later
    test_X = test_X.astype(np.float32)
    test_X = np.nan_to_num(test_X)
    test_Y = np.nan_to_num(test_Y)
    
    # scale & predict
    test_X_transf = scaler.transform(test_X)
    test_probas = forest_clf.predict_proba(test_X_transf)
        
    test_rmse = np.sqrt(mean_squared_error(test_Y, test_probas[:, 1]))
    print('\n Root Mean Squared Error for the test set:', test_rmse)
    rmse = np.zeros(2)
    rmse[0] = train_rmse
    rmse[1] = test_rmse

    # make & check classific.
    test_predictions = np.zeros(test_Y.size)
    correct_test = 0
    for i in range(test_Y.size):
        if test_probas[i, 1] >= 0.5:
            test_predictions[i] = 1
        if test_predictions[i] == test_Y[i]:
            correct_test += 1

    print('\n Amount of correctly classified test sources:', correct_test)

    # test accuracy
    accuracy_test = correct_test/test_Y.size
    print('\n Testing Accuracy:', accuracy_test)
    acc = np.zeros(2)
    acc[0] = accuracy_train
    acc[1] = accuracy_test

    return rmse, acc, forest_clf, scaler


train_count = 2
# run multiple times to calculate errors
rmse_train = np.zeros(train_count)
acc_train = np.zeros(train_count)
rmse_test = np.zeros(train_count)
acc_test = np.zeros(train_count)

for i in range(train_count):
    rmse_i, acc_i, forest_clf, scaler = forest_train(sources_assoc, nTrain=i,
                                                     augmentation=None)
    rmse_train[i] = rmse_i[0]
    rmse_test[i] = rmse_i[1]
    acc_train[i] = acc_i[0]
    acc_test[i] = acc_i[1]

print('\nFinal Results: \n')
print('Train Mean acc.:', np.mean(acc_train), 'std. dev.', np.std(acc_train))
print('Test Mean acc: ', np.mean(acc_test), 'std dev', np.std(acc_test))


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
                          bcu_index.iloc[i]['ASSOC1'],
                          bcu_probas[i, 1]])
    elif bcu_index.iloc[i]['BZ_Class'] == 'bll':
        bll_list.append([bcu_index.iloc[i]['Source_Name'],
                         bcu_index.iloc[i]['ASSOC1'],
                         bcu_probas[i, 0]])


# %% SCATTER PLOTS
def scatter_plots(sources_pd):
    file2 = 'logs\\20210101-190513_test\\final_predictions.npy'
    pred = np.load(file2)
    new_class_data = {'BZ_Class': pred[:, 12-2],
                      'Source_Name': pred[:, 0],
                      'Assoc': pred[:, 1],
                      'RA_J2000': pred[:, 2],
                      'DE_J2000': pred[:, 3],
                      'Variability_Index': pred[:, 4],
                      'PL_Flux_Density': pred[:, 5],
                      'Flux1000': pred[:, 6],
                      'Pivot_Energy': pred[:, 7],
                      'Signif_Avg': pred[:, 8],
                      'PL_Index': pred[:, 9],
                      'Class_Prob': pred[:, 12-1]}

    new_class = pd.DataFrame(new_class_data, columns=['BZ_Class',
                                                      'Source_Name', 'Assoc',
                                                      'RA_J2000', 'DE_J2000',
                                                      'Variability_Index',
                                                      'PL_Flux_Density',
                                                      'Flux1000',
                                                      'Pivot_Energy',
                                                      'Signif_Avg', 'PL_Index',
                                                      'Class_Prob'])

    new_class['RA_J2000'] = new_class['RA_J2000'].astype(float)
    new_class['DE_J2000'] = new_class['DE_J2000'].astype(float)
    new_class['Variability_Index'] = new_class['Variability_Index'].astype(float)
    new_class['PL_Flux_Density'] = new_class['PL_Flux_Density'].astype(float)
    new_class['Flux1000'] = new_class['Flux1000'].astype(float)
    new_class['Pivot_Energy'] = new_class['Pivot_Energy'].astype(float)
    new_class['Signif_Avg'] = new_class['Signif_Avg'].astype(float)
    new_class['PL_Index'] = new_class['PL_Index'].astype(float)

    bll_new = new_class[new_class["BZ_Class"] == 'bll']
    fsrq_new = new_class[new_class["BZ_Class"] == 'fsrq']
    bll = sources_pd[sources_pd["BZ_Class"] == 0]
    fsrq = sources_pd[sources_pd["BZ_Class"] == 1]
    # bcu = sources_pd[sources_pd["BZ_Class"] == 2]

    # Scatter PL_Ind vs. Variab
    plt.scatter(np.log10(bll["PL_Flux_Density"]),
                np.log10(bll["Variability_Index"]),
                label='BL Lacs (4FGL)', color='green', edgecolors='none',
                alpha=0.5, marker='.')
    plt.scatter(np.log10(fsrq["PL_Flux_Density"]),
                np.log10(fsrq["Variability_Index"]), label='FSRQs (4FGL)',
                color='red', edgecolors='none', alpha=0.5, marker='.')
    plt.scatter(np.log10(bll_new["PL_Flux_Density"]),
                np.log10(bll_new["Variability_Index"]), label='BL Lacs (NN)',
                color='orange', edgecolors='none', alpha=0.5, marker='.')
    plt.scatter(np.log10(fsrq_new["PL_Flux_Density"]),
                np.log10(fsrq_new["Variability_Index"]), label='FSRQs (NN)',
                color='purple', edgecolors='none', alpha=0.5, marker='.')
    plt.xlabel('$log_{10}$ Differential Flux at Pivot Energy in $cm^{-2}MeV^{-1}s^{-1}$')
    plt.ylabel('$log_{10}$ of Variability Index')
    plt.legend()
    plt.savefig('Flux_Density_Variab_scatter.png', dpi=300)
    plt.show()

    # Pivot_E vs. Signif
    plt.scatter(np.log10(bll["Pivot_Energy"]), np.log10(bll["Signif_Avg"]),
                label='BL Lacs (4FGL)', color='green', edgecolors='none',
                alpha=0.5, marker='.')
    plt.scatter(np.log10(fsrq["Pivot_Energy"]), np.log10(fsrq["Signif_Avg"]),
                label='FSRQs (4FGL)', color='red', edgecolors='none',
                alpha=0.5, marker='.')
    plt.scatter(np.log10(bll_new["Pivot_Energy"]),
                np.log10(bll_new["Signif_Avg"]),
                label='BL Lacs (NN)', color='orange', edgecolors='none',
                alpha=0.5, marker='.')
    plt.scatter(np.log10(fsrq_new["Pivot_Energy"]),
                np.log10(fsrq_new["Signif_Avg"]),
                label='FSRQs (NN)', color='purple', edgecolors='none',
                alpha=0.5, marker='.')
    plt.xlabel('log10 of Pivot Energy in MeV')
    plt.ylabel('log10 of Source Significance in $ \sigma$')
    plt.legend()
    plt.savefig('Pivot_E_Signif_scatter.png', dpi=300)
    plt.show()

    # int flux vs. latitude
    plt.scatter(np.log10(bll["Flux1000"]), bll["PL_Index"],
                label='BL Lacs (4FGL)', color='green', edgecolors='none',
                alpha=0.5, marker='.')
    plt.scatter(np.log10(fsrq["Flux1000"]), fsrq["PL_Index"],
                label='FSRQs (4FGL)', color='red', edgecolors='none',
                alpha=0.5, marker='.')
    plt.scatter(np.log10(bll_new["Flux1000"]), bll_new["PL_Index"],
                label='BL Lacs (NN)', color='orange', edgecolors='none',
                alpha=0.5, marker='.')
    plt.scatter(np.log10(fsrq_new["Flux1000"]), fsrq_new["PL_Index"],
                label='FSRQs (NN)', color='purple', edgecolors='none',
                alpha=0.5, marker='.')
    plt.xlabel('$log_{10}$ of Integral photon flux in $cm^{-2}s^{-1}$')
    plt.ylabel('Power Law Photon Index')
    plt.legend()
    plt.savefig('Flux_PL_Index_scatter.png', dpi=300)
    plt.show()

scatter_plots(sources_pd)

# %% TIMER
stop = timeit.default_timer()
print('\n Execution Time: ', stop - start, 's')


# %% RECYCLING BIN

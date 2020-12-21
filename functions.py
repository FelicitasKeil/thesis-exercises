#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from catalog import catalog
import tensorflow.keras as K
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from scipy.stats import wasserstein_distance
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import sys
import time
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from astropy.io import fits
import math


def process_input():
    class Args:
        augmentation_style = None
        network = 'DNN'
        catalog = 'gll_psc_v26.fit'
        norm_axis = 0
        xVal = False
        log_xVal = False
        log = False
        log_full = False
        savetag = 'test'
        seed = None
        add_static = None
        batchsize = 256
        learning_rate = 1e-2
        series_layers = 1
        nodes_per_layer = 16
        classification_layers = 2
        class_nodes = 8
        series = 'both'

    parser = ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--augmentation_style', default=None,
                        help="""One of ['repeat', 'noise'],
                        defines augmentation style.\n
                        default: None""")
    parser.add_argument('--network', default='DNN',
                        help="""One of ['RNN', 'DNN', 'LSTM', 'BID'], defining
                        the architecture to be used.\n
                        default: RNN""")
    parser.add_argument('--catalog', default='gll_psc_v26.fit',
                        help="""Path to the Fermi LAT catalog fit file.\n
                        default: 'gll_psc_v26.fit'""")
    parser.add_argument('--norm_axis', default=0,
                        help="""Axis along which the standard normalization
                        should be performed. 0 gives the pixel wise, (0, 2) the
                        row wise normalization.\n
                        default: 0""")
    parser.add_argument('--xVal', action='store_false',
                        help="""Set this flag to skip cross validation
                        training""")
    parser.add_argument('--log_xVal', action='store_true',
                        help="""Set this flag to store training curves of the
                        cross validation in a log directory""")
    parser.add_argument('--log', action='store_false',
                        help="""Set this flag to NOT store training curves
                        of the final training""")
    parser.add_argument('--log_full', action='store_true',
                        help="""Set this flag to enable --log and
                        --log_xVal""")
    parser.add_argument('--savetag', default='test',
                        help="""Give a tag that is added to all saved
                        information to find results more easily.\n
                        default: test""")
    parser.add_argument('--seed', default=None, type=int,
                        help="""Set a numpy seed for reproducibility of the
                        random data augmentation""")
    parser.add_argument('--add_static', default=None, nargs='+',
                        help="""Add given list of static features, e.g.
                        'Variability_Index' or 'LP_SigCurv'""")
    parser.add_argument('--batchsize', default=64, type=int,
                        help="""Set the batchsize for training.\n
                        Default: 64""")
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help="""Initial learning rate for training.\n
                        Default: 1e-3""")
    parser.add_argument('--series_layers', default=1, type=int,
                        help="""Number of layers for each individual series.\n
                        Default: 1""")
    parser.add_argument('--nodes_per_layer', default=8, type=int,
                        help="""Number of nodes for individual layers.\n
                        Default: 8""")
    parser.add_argument('--classification_layers', default=2, type=int,
                        help="""Number of classification layers after series
                        layers\n
                        Default: 2""")
    parser.add_argument('--class_nodes', default=8, type=int,
                        help="""Number of nodes in classification layers\n
                        Default: 8""")
    parser.add_argument('--series', default='energy',
                        help="""One of ['time', 'energy', 'both'], indicating
                        which series data to use from the catalog.\n
                        Default: 'both'""")

    args, _ = parser.parse_known_args()
    if any(_):
        print('\n\nUnknown arguments provided, using defaults')
        args = Args()

    if args.series == 'both':
        args.nSeries = 2
    elif args.series in ['time', 'energy']:
        args.nSeries = 1
    else:
        raise SystemExit('Not a valid series argument')

    # Get a timestamp which can be associated uniquely with a run of the script
    args.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.log_full:
        args.log = True
        args.log_xVal = True

    if args.seed is None:
        args.seed = int(datetime.now().strftime("%m%d%H%M%S"))

    # Set a random seed for numpy, for reproducibility
    np.random.seed(args.seed)

    if args.log or args.log_xVal:
        args.logdir = f'logs/{args.timestamp}_{args.savetag}'
        os.mkdir(args.logdir)
        with open(args.logdir + '/args.txt', 'w') as f:
            f.write('\n'.join(sys.argv[1:]))
            f.write(f'\nnetwork:\n{args.network}\n')
            f.write(f'\ncatalogue:\n{args.catalog}\n')
            f.write(f'\nstatic variables:\n{args.add_static}\n')
            f.write(f'\nlearning rate:\n{args.learning_rate}\n')
            f.write(f'\nlayers for series variables:\n{args.series_layers}\n')
            f.write(f'\nnodes per layer:\n{args.nodes_per_layer}\n')
            f.write(f'\nseries variables:\n{args.series}\n')
            if '--seed' not in sys.argv[1:]:
                f.write(f'\nseed: \n{args.seed}\n')
    else:
        args.logdir = None

    args.bidirectional = False
    return args

# %% Different catalogues


def merge_catalogs(cat_default, cat_new):
    """ Add 4LAC AGN catalogue features to the data.

    Parameters
    ----------
    cat_default : original (4FGL) catalogue
    cat_new : new (4LAC) catalogue with additional features

    Returns
    -------
    merged catalogue (only 1-dim features)

    """
    default_cat = catalog(cat_default, 1)
    default_cat()

    default_cat_pd = default_cat.pdTable

    with fits.open(cat_new) as data:
        new_sources_pd = pd.DataFrame(data[1].data)

    drop_cols = [col for col in new_sources_pd if col in default_cat_pd]
    drop_cols.remove('Source_Name')
    drop_cols.append('CLASS')

    new_sources_pd = new_sources_pd.drop(drop_cols, axis=1)
    default_cat_trimmed = default_cat_pd.apply(lambda x: x.str.strip()
                                               if x.dtype == "object" else x)

    merged_cat = default_cat_trimmed.merge(new_sources_pd, how="outer",
                                           on='Source_Name')
    merged_cat = merged_cat[merged_cat.GLON.notnull()]

    return merged_cat


def get_cross_catalog_labels(cat_old, cat_new):
    old_catalog = catalog(cat_old, 1)  # open old cat w. catalog.py
    old_catalog()

    one_dim_data = old_catalog.pdTable

    # add column: True if source unknown
    unknown = [True if x.strip() in ['', 'UNK', 'unk']
               else False for x in one_dim_data['CLASS1']]
    one_dim_data['unknown'] = unknown

    one_dim_data = one_dim_data.loc[one_dim_data['unknown']]
    # for key in multi_dim_data.keys():
    #     multi_dim_data[key] = np.array(multi_dim_data[key])[unknwon]

    new_catalog = catalog(cat_new, 1)
    new_catalog()

    future_label = [get_future_label(new_catalog, source)
                    for source in one_dim_data['Source_Name']]

    return np.array(future_label)


def get_future_label(newcat, sourcename, positives=["PSR", "psr"],
                     negatives=["FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu",
                                "CSS", "css", "RDG", "rdg", "NLSY1", "nlsy1",
                                "agn", "ssrq", "sey"]):
    """Create a crossmatch between old and new catalog.

    Args:
        newcat: new catalog with future classification
        sourcename: name of the source in the old catalog
        positives (list, optional): Positive class names.
            Defaults to ["FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG",
                        "rdg", "NLSY1", "nlsy1", "agn", "ssrq", "sey"].
        negatives (list, optional): Negative class names.
            Defaults to ["PSR", "psr"].
    Returns:
        1 if future classification is in positives
        0 if future classification is in negatives
        nan otherwise
    """
    newnames = newcat.source_name
    newcrossmatch = newcat.cat_table['ASSOC_FGL']
    index = -99
    for i in range(len(newcrossmatch)):
        # implement faster option; only trial as strings have different spacing
        if(sourcename in newcrossmatch[i] and
           newcat.cat_table['CLASS1'][i].strip() not in newcat.class_unk):
            index = i
            break
    if(index > 0):
        if(newcat.cat_table['CLASS1'][index].strip() in positives):
            return 1
        elif(newcat.cat_table['CLASS1'][index].strip() in negatives):
            return 0
        else:
            return np.nan
    else:
        return np.nan


# %% Prepare Data

def get_multi_series_data(cat, add_static=None, series='both',
                          label='PSR-AGN'):
    # Load catalog
    mycatalog = catalog(cat, 1)
    mycatalog()

    # Extract multidimensional and one dimensional features
    multi_dim_data = load_multidimensional_catalog(mycatalog)
    # one_dim_data_old = mycatalog.pdTable
    one_dim_data = merge_catalogs('gll_psc_v26.fit', 'table_4LAC.fits')

    # Label PSRs (0) and AGNs (1), other classes are assigned NaN
    if label == 'PSR-AGN':
        labels = get_label(one_dim_data['CLASS1'])
    # Label MSPs (0) and YNGs (1), other classes are assigned NaN
    elif label == 'MSP-YNG':
        labels = np.array([0 if x == 'MSP' else 1 if x == 'YNG'
                           else np.nan for x in mycatalog.psr_subclass])
    elif label == 'BLL-FSRQ':
        labels = get_label(one_dim_data['CLASS1'],
                           negatives=["BLL", "bll"],
                           positives=["FSRQ", "fsrq"])
    elif label == 'unlabeled':
        labels = [0 if x.strip() in ['', 'UNK', 'unk']
                  else np.nan for x in one_dim_data['CLASS1']]
    else:
        raise SystemExit('Not a valid argument for "label"')

    one_dim_data['Labels'] = np.array(labels)
    multi_dim_data['Labels'] = np.array(labels)

    # Get indices of NaN values in the labels
    not_labeled = one_dim_data.Labels.isna()

    # Remove all entries without label
    multi_dim_data_labeled = {}
    one_dim_data_labeled = {}
    for key in multi_dim_data.keys():
        multi_dim_data_labeled[key] = np.array(
            multi_dim_data[key][~not_labeled])

    for key in one_dim_data.keys():
        one_dim_data_labeled[key] = np.array(one_dim_data[key][~not_labeled])

    # Try stacking features, fails for 3FGL
    try:
        hist_series = np.stack((multi_dim_data_labeled['Flux_History'],
                                multi_dim_data_labeled['Sqrt_TS_History']),
                               axis=-1)
        # hist_series = multi_dim_data_labeled['Flux_History'][..., np.newaxis]
        band_series = np.stack((multi_dim_data_labeled['Flux_Band'],
                                multi_dim_data_labeled['Sqrt_TS_Band']),
                               axis=-1)
        # band_series = multi_dim_data_labeled['Flux_Band'][..., np.newaxis]
    except:
        # Construct series data for 3FGL catalog
        print("\n\nUsing 3FGL\n\n")
        fluxes = ['100_300', '300_1000', '1000_3000',
                  '3000_10000', '10000_100000']
        flux_band = [one_dim_data_labeled[f'Flux{x}'] for x in fluxes]
        sqrt_ts_band = [one_dim_data_labeled[f'Sqrt_TS{x}'] for x in fluxes]
        hist_series = np.stack((np.transpose(flux_band),
                                np.transpose(sqrt_ts_band)),
                               axis=-1)
        band_series = np.expand_dims(multi_dim_data_labeled['Flux_History'],
                                     -1)
        # band_series = np.sort(band_series, axis=1)

    if add_static is not None:
        print('Adding static features', add_static)

        stat_features = [one_dim_data_labeled[key] for key in add_static]
        if len(add_static) == 1:
            stat_features = np.expand_dims(stat_features, 0)
            stat_features = np.squeeze(stat_features, 0)

        stat_features = np.transpose(stat_features)
        for i in range(len(stat_features[0])):
            median = np.nanmedian(stat_features, axis=0)
            stat_features[np.isnan(stat_features[:, i])] = median[i]
            stat_features[np.where(stat_features[:, i] == 0)[0]] = median[i]

        if series == 'both':
            return hist_series, band_series, stat_features, \
                multi_dim_data_labeled['Labels'], \
                one_dim_data_labeled['Source_Name']
        elif series == 'time':
            return hist_series, stat_features, \
                multi_dim_data_labeled['Labels'], \
                one_dim_data_labeled['Source_Name']
        elif series == 'energy':
            return band_series, stat_features, \
                multi_dim_data_labeled['Labels'], \
                one_dim_data_labeled['Source_Name']

    else:
        if series == 'both':
            return hist_series, band_series, \
                multi_dim_data_labeled['Labels'], \
                one_dim_data_labeled['Source_Name']
        elif series == 'time':
            return hist_series, multi_dim_data_labeled['Labels'], \
                one_dim_data_labeled['Source_Name']
        elif series == 'energy':
            return band_series, multi_dim_data_labeled['Labels'], \
                one_dim_data_labeled['Source_Name']

    raise SystemExit('Something in getting data went wrong')


def load_multidimensional_catalog(cat):
    """Given a source catalog (Silvia's class), this module reads all features
    with multiple dimensions.

    Args:
        cat (catalog): catalog object created with Silvia's catalog class

    Returns:
        data (dict): dictionary with feature names as keys & corresponding data
                as values
    """

    data = {}
    for feature in cat.features_names:
        if len(np.shape(cat.cat_table[feature])) > 1:
            data[feature] = cat.cat_table[feature]

    return data


def get_label(classes,
              negatives=["FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu",
                         "CSS", "css", "RDG", "rdg", "NLSY1", "nlsy1",
                         "agn", "ssrq", "sey"],
              positives=["PSR", "psr"]):
    """Return labels. 1 for classes entries contained in positives, 0 for those
    in negatives and NaN for unidentified classe entries.

    Args:
        classes (list): list of class labels
        positives (list, optional): Positive class names.
            Defaults to ["FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG",
                        "rdg", "NLSY1", "nlsy1", "agn", "ssrq", "sey"].
        negatives (list, optional): Negative class names.
            Defaults to ["PSR", "psr"].

    Returns:
        list: List containing labels for classes
    """
    for n, i in enumerate(classes):
        if pd.isnull(i):
            classes[n] = ''

    labels = [1 if x.strip() in positives else 0 if x.strip()
              in negatives else np.nan for x in classes]
    return np.array(labels)


def train_split(data, labels, test_size=0.3, **kwargs):
    """Split the dataset conserving the relative amount of the classes.

    Args:
        data (nSources, timesteps, features): Array containing list of features
        labels (nSources): List containg the labels corresponding to data
        test_size (float, optional): Relative amount of training data.
                Defaults to 0.3.

    Returns:
        (tuple): (train data, train labels)
        (tuple): (test data, test labels)
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, **kwargs)
    for train_indices, test_indices in split.split(np.zeros(len(data[0])),
                                                   labels):
        train_data = tuple(x[train_indices] for x in data)
        test_data = tuple(x[test_indices] for x in data)
        train_labels = K.utils.to_categorical(labels[train_indices], 2)
        test_labels = K.utils.to_categorical(labels[test_indices], 2)

    # print(f"\nTraining shapes: {tuple(x.shape for x in train_data)}")
    # print(f"Testing shapes: {tuple(x.shape for x in test_data)}\n")

    return (train_data, train_labels), (test_data, test_labels)


def preprocessing(data, mean=None, std=None, test=False, axis=0):
    """Perform preprocessing steps with the source data, i.e. shift mean to 0
    and divide by standard deviation

    Args:
        sources (nSources, timesteps, features): List of sources with features
        mean (1, timesteps, features): When preprocessing test data, provided
                mean array
        std (1, timesteps, features): When preprocessing test data, provided
                std array
        test (bool): If set true, use provided mean and std arrays for
                normalization

    Returns:
        sources (nSources, timesteps, features): sources preprocessed

        if test == False:
        mean (1, timesteps, features): mean source image
        std (1, timesteps, features)
    """
    if test:
        if mean is None:
            assert 'Provide mean for normalization'
        if std is None:
            assert 'Provide std for normalization'

    else:
        mean = tuple(np.mean(x, axis=0, keepdims=True) for x in data)
        std = tuple(np.std(x, axis=0, keepdims=True) for x in data)

    prep = tuple((dat - mean) / std for (dat, mean, std)
                 in zip(data, mean, std))
    if test:
        return prep
    else:
        return prep, mean, std


def lr_schedule(epoch):
    min_lr = 6e-4
    max_lr = 5e-3
    stepsize = 10

    if epoch // (2 * stepsize) != 0:
        min_lr /= 2 ** (epoch // (2 * stepsize))
        max_lr /= 2 ** (epoch // (2 * stepsize))
    rising = np.linspace(min_lr, max_lr, stepsize)

    learning_rates = np.append(rising, rising[::-1])

    return learning_rates[epoch % (2 * stepsize)]

# %% Train & test network


def cross_validation_training(gen_class, data, labels, nFolds=10, nSeries=2,
                              class_weights=None, augmentation_style=None,
                              logdir=None, batchsize=512, epochs=300,
                              network_params={}):
    """This function performs kFold cross validation.

    Args:
        gen_class (function): Function to generate the classifier
        data (tuple): Tuple of training data
        labels (ndarray): One hot encoded training labels
        nFolds (int, optional): [description]. Defaults to 10.
        class_weights (dir, optional): Directory of class weigths for training.
                                    Defaults to None.
        augmentation_style (str, optional): Augmentation style to be used for
                                    augmenting training data. Defaults to None.
        logdir (str, optional): Path to a directory in which to store logs.
                                    Defaults to None.

    Returns:
        dic: Dictionary containing training and validation results
    """

    print(f'{"="*40}\nStarted {nFolds} fold cross validation\n')
    skfolds = StratifiedKFold(n_splits=nFolds, shuffle=True)
    results_test = []
    results_train = []
    fold = 0
    for train_index, val_index in skfolds.split(np.zeros(len(data[0])),
                                                labels.argmax(1)):
        start_fold = time.time()
        fold += 1
        print(f'Fold {fold}')

        # Create callbacks, log with tensorboard, if logdir is given
        callbacks = [K.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True, verbose=1)]
        # callbacks.append(K.callbacks.LearningRateScheduler(lr_schedule))
        if logdir is not None:
            log_fold = f'{logdir}/fold{fold}'
            callbacks.append(K.callbacks.TensorBoard(log_dir=log_fold))
        else:
            log_fold = None

        # Seperate training and validation set for training
        train_set = tuple(x[train_index] for x in data)
        validation_set = tuple(x[val_index] for x in data)

        train_labels = labels[train_index]
        validation_labels = labels[val_index]

        # Perform data augmentation on training data
        if augmentation_style is not None:
            aug_x, aug_y = data_augmentation(
                train_set, train_labels, style=augmentation_style)

            train_x = tuple(np.append(x, y, axis=0)
                            for x, y in zip(train_set, aug_x))
            train_y = np.append(train_labels, aug_y, axis=0)

            indices = np.random.permutation(len(train_x[0]))
            train_x = tuple(x[indices] for x in train_x)
            train_y = train_y[indices]
        else:
            train_x = train_set
            train_y = train_labels

        # Create a new classifier and train it
        classifier = gen_class(train_set, **network_params)

        fit = classifier.fit(train_x, train_y,
                             epochs=epochs,
                             batch_size=batchsize,
                             verbose=0,
                             class_weight=class_weights,
                             callbacks=callbacks,
                             validation_data=(validation_set,
                                              validation_labels))

        # Get results on current training and validation set
        tmp_results = test_classifier(classifier, train_set, train_labels,
                                      validation_set, validation_labels,
                                      logdir=None)
        # results.append[tmp_results]
        results_train.append(tmp_results[0])
        results_test.append(tmp_results[1])

        # tmp_results = test_classifier(
        #    classifier, validation_set, validation_labels, logdir=None)
        # results_test.append(tmp_results)

        # Clear tf session to remove old model
        K.backend.clear_session()

        print('Fold took {} seconds'.format(int(time.time() - start_fold)))

    # Put together result dictionaries
    results_train_dic = {}
    results_test_dic = {}

    for key in results_train[0]:
        results_train_dic[key] = [dic[key] for dic in results_train]

    for key in results_test[0]:
        results_test_dic[key] = [dic[key] for dic in results_test]

    # Print results of cross validation
    print(classifier.summary())
    print('\n\nCross validation results')
    print('On training sets:')
    for key in results_train_dic:
        mean = np.mean(results_train_dic[key])
        std = np.std(results_train_dic[key])
        print(f'{key:<12}:\t{mean:.4f}\t{std:.1e}')

    print('\nOn test sets:')
    for key in results_test_dic:
        mean = np.mean(results_test_dic[key])
        std = np.std(results_test_dic[key])
        print(f'{key:<12}:\t{mean:.4f}\t{std:.1e}')

    print('='*60)

    # Save the cross validation results in the log directory
    if logdir is not None:
        np.savez(logdir+'/results_test', **results_test_dic)
        np.savez(logdir+'/results_train', **results_train_dic)

    return {'train': results_train_dic, 'val': results_test_dic}


def test_classifier(classifier, train_X, train_Y, test_X, test_Y, logdir):
    def save_roc(prediction, label):
        eff_0, eff_1, _ = roc_curve(label, prediction[:, 1],
                                    drop_intermediate=False)

        i = 0
        while True:
            roc_file = f'{logdir}/ROC_{i}.npz'
            if os.path.isfile(roc_file):
                i += 1
            else:
                break

        np.savez(roc_file, eff_pos=eff_1, eff_neg=eff_0)

    def threshold_scan(predictions, labels):
        thresholds = np.linspace(0, 1, 100)  # sorted(predictions[:, 0])
        tot_accs = np.empty(len(thresholds))
        neg_accs = np.empty(len(thresholds))
        pos_accs = np.empty(len(thresholds))
        f1_scores = np.empty(len(thresholds))

        for ind, thresh in enumerate(thresholds):
            preds = np.array([0 if x > thresh else 1
                              for x in predictions[:, 0]])
            tot_accs[ind] = np.mean(preds == labels)
            neg_accs[ind] = np.mean(preds[labels == 0] == 0)
            pos_accs[ind] = np.mean(preds[labels == 1] == 1)
            f1_scores[ind] = f1_score(preds, labels)

        acc = np.mean(predictions.argmax(1) == labels)
        hi_index = tot_accs.argmax()
        hi_acc = tot_accs[hi_index]
        hi_thresh = thresholds[hi_index]

        eq_index = np.argmin(np.abs(neg_accs - pos_accs))
        eq_acc = tot_accs[eq_index]
        eq_thresh = thresholds[eq_index]

        training_results['Acc'] = acc
        training_results['Hi_acc'] = hi_acc
        training_results['Pos_acc'] = pos_accs[hi_index]
        training_results['Neg_acc'] = neg_accs[hi_index]
        training_results['F1'] = f1_scores[hi_index]
        training_results['Eq_acc'] = eq_acc
        training_results['AUC'] = roc_auc_score(labels, predictions[:, 1])
        return hi_thresh, eq_thresh

    training_results = {}
    testing_results = {}

    train_y = train_Y.argmax(1)
    test_y = test_Y.argmax(1)

    train_pred = classifier.predict(train_X)
    test_pred = classifier.predict(test_X)

    if logdir is not None:
        save_roc(test_pred, test_y)

    hi_thresh, eq_thresh = threshold_scan(train_pred, train_y)
    hi_preds = np.array([0 if x > hi_thresh else 1 for x in test_pred[:, 0]])
    eq_preds = np.array([0 if x > eq_thresh else 1 for x in test_pred[:, 0]])

    testing_results['Acc'] = np.mean(test_pred.argmax(1) == test_y)
    testing_results['Hi_acc'] = np.mean(hi_preds == test_y)
    testing_results['Pos_acc'] = np.mean(hi_preds[test_y == 1] == 1)
    testing_results['Neg_acc'] = np.mean(hi_preds[test_y == 0] == 0)
    testing_results['F1'] = f1_score(hi_preds, test_y)
    testing_results['Eq_acc'] = np.mean(eq_preds == test_y)
    testing_results['AUC'] = roc_auc_score(test_y, test_pred[:, 1])

    return training_results, testing_results


def old_test_classifier(classifier, test_X, labels_oh, logdir=None):
    """Test a given classifier and give the loss, acc and acc by class

    Args:
        classifier (keras.Model): Classification model
        test_X (ndarray): Array containing the test data
        labels_oh ([type]): Array containing one hot encoded labels

    Returns:
        dict: Dictionary containing the evaluated performance measures
    """
    labels = labels_oh.argmax(1)
    loss, acc = classifier.evaluate(test_X, labels_oh, verbose=0)

    predictions = classifier.predict(test_X)

    if logdir is not None:
        eff_neg, eff_pos, _ = roc_curve(
            labels, predictions[:, 1], drop_intermediate=False)
        roc_file = f'{logdir}/ROC_0.npz'
        i = 0
        while True:
            i += 1
            if os.path.isfile(roc_file):
                roc_file = f'{logdir}/ROC_{i}.npz'
            else:
                break
        np.savez(roc_file, eff_pos=eff_pos, eff_neg=eff_neg)

    auc = roc_auc_score(labels, predictions[:, 1])

    pos_pred = predictions[labels == 1]
    neg_pred = predictions[labels == 0]

    pos_acc = np.mean(pos_pred.argmax(1) == 1)
    neg_acc = np.mean(neg_pred.argmax(1) == 0)

    tot_accs = np.empty(len(predictions))
    neg_accs = np.empty(len(predictions))
    pos_accs = np.empty(len(predictions))
    for ind, thresh in enumerate(sorted(predictions[:, 0])):
        preds = np.array([0 if x > thresh else 1 for x in predictions[:, 0]])
        tot_accs[ind] = np.mean(preds == labels)
        neg_accs[ind] = np.mean(preds[labels == 0] == 0)
        pos_accs[ind] = np.mean(preds[labels == 1] == 1)

    diff_acc = abs(neg_accs - pos_accs)
    min_ind = np.argmin(diff_acc)
    max_ind = np.argmax(tot_accs)

    eq_acc = (pos_accs[min_ind] + neg_accs[min_ind]) / 2.
    hi_acc = tot_accs[max_ind]

    results = {'Loss': loss, 'Acc': acc, 'Pos_Acc': pos_acc,
               'Neg_Acc': neg_acc, 'Eq_Acc': eq_acc,
               'Hi_Acc': hi_acc, 'AUC': auc}

    # if not logdir is None:
    #     np.savez(f'{logdir}/Accs',
    #              tot_accs=tot_accs,
    #              pos_accs=pos_accs,
    #              neg_accs=neg_accs,
    #              thresh=sorted(predictions[:, 0]))

    return results


# %% Create new data points


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
        augmented = tuple(x[rand_inds] for x in class_data)
        return augmented

    def noisy_aug(class_data, nAug):
        if nAug <= 0:
            return None
        rand_inds = np.random.choice(len(class_data[0]), nAug)
        augmented = tuple(x[rand_inds] + np.random.normal(
            loc=0., scale=1e-2, size=x[rand_inds].shape) for x in class_data)
        return augmented

    def smote_lite_aug(class_data, nAug):
        if nAug <= 0:
            return None
        indices = np.random.choice(len(class_data[0]), (nAug, 2))
        # create point on the line between two sources in feature space
        augmented = tuple(np.array([x[i] + np.random.uniform() * (x[j] - x[i])
                                    for i, j in indices]) for x in class_data)

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
            distances = np.zeros((class_data[0].shape[0],
                                  class_data[0].shape[0]))
            for data in class_data:
                data = np.reshape(data, (data.shape[0], -1))
                distances += distance_matrix(data, data)

            return distances

        start = time.time()
        dists = calc_distances()
        print(int(time.time() - start))
        exit()
        dists = np.mean(dists, axis=0)

        augmented = []
        for ind1, data in enumerate(class_data):
            top_k_nn = np.argsort(dists[ind1])[1:6]
            nn_inds = np.random.choice(top_k_nn, augFactor - 1)

            for nn_ind in nn_inds:
                augmented.append(
                    data + np.random.uniform() * (class_data[nn_ind] - data)
                )

        return np.array(augmented)

    print(f'\nUsing augmented data, style: {style}\n')
    if labels.shape[-1] == 2:
        labels = labels.argmax(-1)

    nPos = labels.sum()
    nNeg = len(labels) - nPos
    if nPerClass is None:
        nPerClass = np.max([nPos, nNeg])

    pos_data = tuple(x[labels == 1] for x in data)
    neg_data = tuple(x[labels == 0] for x in data)

    if style == 'noise':
        pos_aug = noisy_aug(pos_data, nPerClass - nPos)
        neg_aug = noisy_aug(neg_data, nPerClass - nNeg)
    elif style == 'repeat':
        pos_aug = repeat_aug(pos_data, nPerClass - nPos)
        neg_aug = repeat_aug(neg_data, nPerClass - nNeg)
    elif style == 'smote lite':
        pos_aug = smote_lite_aug(pos_data, nPerClass - nPos)
        neg_aug = smote_lite_aug(neg_data, nPerClass - nNeg)
    elif style == 'smote':
        pos_aug = smote_aug(pos_data, nPerClass - nPos)
        neg_aug = smote_aug(neg_data, nPerClass - nNeg)
    else:
        sys.exit("No supported augmentation style")

    if pos_aug is None:
        augmented = neg_aug
        augmented_y = np.zeros(len(neg_aug[0]))
    elif neg_aug is None:
        augmented = pos_aug
        augmented_y = np.zeros(len(pos_aug[0]))
    else:
        augmented = tuple(np.append(x, y, axis=0)
                          for x, y in zip(pos_aug, neg_aug))
        augmented_y = np.append(
            np.ones(len(pos_aug[0])), np.zeros(len(neg_aug[0])))

    return augmented, K.utils.to_categorical(augmented_y, 2)


if __name__ == '__main__':
    pass

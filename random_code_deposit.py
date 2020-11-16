#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:49:02 2020

@author: felicitaskeil
"""
from sklearn.pipeline import Pipeline
from astropy.table import Table

data = mycatalog.pdTable

data["CLASS1"].value_counts()
agns = ["FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu",
        "RDG", "rdg", "NLSY1", "nlsy1", "agn", "ssrq", "sey"]
psrs = ["PSR", "psr"]


data["Label"] = [1 if (x in agns) else 0 if (x in psrs)         #Label AGNs 1, PSRs 0, others NaN
                 else np.nan for x in data["CLASS1"].str.strip()]



data.replace(r'^\s*$', np.nan, regex=True, inplace=True)        #replace empty entries w. NaN

for feature in data.columns:                                    #delete features w. empty entries
    if data[feature].isna().sum() > 0.05 * len(data[feature]):
        data.drop(feature, axis=1, inplace=True)
        
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(data, data["Label"]):
    train_set = data.iloc[train_index]
    test_set = data.iloc[test_index]


def convert_multidimensions(cat):
    data = {}
    for feature in cat.features_names:
        if len(np.shape(cat.cat_table[feature])) > 1:
            data[feature] = cat.cat_table[feature]

    return data

multidim = convert_multidimensions(mycatalog)
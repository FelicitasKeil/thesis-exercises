#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:49:02 2020

@author: felicitaskeil
"""
from sklearn.pipeline import Pipeline
from astropy.table import Table
import pandas as pd
from functions import functions

# create first DataFrame
df1 = pd.DataFrame({'rating': [90, 85, 82, 88, 94, 90, 76, 75],
                    'points': [25, 20, 14, 16, 27, 20, 12, 15],
                    'label': [1, 2, 3, 4, 5, 6, 7, 8]})
df2 = pd.DataFrame({'assists': [5, 7, 7, 8, 5, 7],
                    'rebounds': [11, 8, 10, 6, 6, 9],
                    'label': [6, 3, 2, 7, 5, 10]})
print(df1, df2)

df_join = df1.merge(df2, how="outer", left_on='label', right_on='label')
print(df_join)





"""
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

#%% functions


    blz_sc=[]
    cat_bll=[]
    cat_fsrq=[]
    for i in range(self.Ns):
        local_name=str(self.pdTable['ASSOC1'][i])
        #if(local_name[4:].strip)

def get_label_blazars(classes,
                      negatives=["BLL", "bll"],
                      positives=["FSRQ", "fsrq"]):


    Parameters
    ----------
    classes : TYPE
        DESCRIPTION.
    negatives : TYPE, optional
        DESCRIPTION. The default is ["BLL", "bll"].
    positives : TYPE, optional
        DESCRIPTION. The default is ["FSRQ", "fsrq"].

    Returns
    -------
    None.

"""  
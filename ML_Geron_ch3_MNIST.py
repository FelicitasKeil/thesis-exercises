# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:09:31 2020

@author: felicitaskeil
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

start = timeit.default_timer()                                  #start timer

print('\n a) Binary Classifier: if digit is a 5 or not')

mnist = fetch_openml('mnist_784', version=1)                    #get handwritten digits
X, y = mnist["data"], mnist["target"]

y=y.astype(np.uint8)                                            #convert string to int

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:] #separate training & test set (smaller train set for speed)

y_train_5 = (y_train==5)                                        #binary classifier 5/non5
y_test_5 = (y_test==5)

scaler = StandardScaler()                                       #Standardising values (Gaussian)
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

sgd_clf = SGDClassifier(random_state=42, max_iter=1500)                        #stochastic gradient descent

'''-------------------------- PLOT EXAMPLE DIGIT -------------------------------------'''

rand = random.randint(0, 60000)
some_digit = X[rand]                                             #chooose example digit to test
some_value = y[rand]
print('Random Digit chosen: ', rand, 'with real value:', y[rand])
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.title('Random Handwritten Digit from MNIST')
plt.axis="off"
plt.show()

'''-------------------------- CROSS VALIDATION --------------------------------------'''

skfolds = StratifiedKFold(n_splits = 3, random_state=42, shuffle=True)

for train_index, test_index in skfolds.split(X_train_scaled, y_train_5): #own cross_val_score fct.
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train_scaled[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train_scaled[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred==y_test_fold)
    
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


'''-------------------------- PRECISION & RECALL -----------------------------------'''

y_scores = cross_val_predict(sgd_clf, X_train_scaled, y_train_5, cv=3,
                             method="decision_function")        #decision scores of k-fold CV

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): #plot precision, recall
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


'''----------------------- RECEIVER OPERATING CHARACTERISTIC (ROC) -------------------'''

forest_clf = RandomForestClassifier(random_state=42)            #use random forest

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):                       #plot ROC curve for SGD
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

y_probas_forest = cross_val_predict(forest_clf, X_train_scaled, y_train_5, cv=3, 
                                    method = "predict_proba")

print('Forest proability: ', y_probas_forest[rand])

y_scores_forest = y_probas_forest[:, 1]                         #use probabilities as score

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")                           #plot the ROC of SGD & RF to compare
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc="lower right")
plt.show()

print('\n ROC area under curve of SGD: ', roc_auc_score(y_train_5, y_scores)) #print area under curve
print('\n ROC_AUC of Forest: ', roc_auc_score(y_train_5, y_scores_forest))

'''-------------------------- MULTIPLE CLASSES -------------------------------------'''

print('\n b) Multiple Classifier: for distinguishing all 10 digits')

forest_clf.fit(X_train_scaled, y_train)                         #digit probabilities with RF
forest_clf.predict([some_digit])
forest_probas = np.asarray(forest_clf.predict_proba([some_digit]))
print('\n All digit probabilities with Random Forest: ', forest_probas)

predicted_class_forest = np.array(forest_probas.argmax())
print('\n Predicted digit from RF: ', predicted_class_forest)


sgd_clf.fit(X_train_scaled, y_train)                            #fit SGD for all 10 digits

some_digit_scores = sgd_clf.decision_function([some_digit])     #get scores for all digits

predicted_class = sgd_clf.classes_[np.argmax(some_digit_scores)] #get predicted label
print('\n Predicted digit from SGD: ', predicted_class)

print('\n CV accuracy score for SGD', cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, 
                                           scoring="accuracy")) #CV score


'''-------------------------- ERROR ANALYSIS ---------------------------------------'''

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)               #plot confusion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.title('Confusion Matrix \n ')
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)                   #stadardize error
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)                               #set diagonal to 0
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)                     #plot confusion matrix without diagonal
plt.title('Confusion Matrix without diagonal \n')
plt.show()

'''-------------------------- TIMER ------------------------------------------------'''

stop = timeit.default_timer()                                   #time program
print('\n Execution Time: ', stop - start, 's') 

''' -----------------------RECYCLING BIN----------------------------------------------
                                           
sgd_clf.fit(X_train, y_train_5)

'''
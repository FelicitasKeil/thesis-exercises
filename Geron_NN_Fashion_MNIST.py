#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:20:40 2020

@author: felicitaskeil
"""
import timeit
start = timeit.default_timer() 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

'''-------------------------- CLASSIFIER --------------------------------------------'''

fashion_mnist = keras.datasets.fashion_mnist                    #load MNIST Data
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0 #separate train & test + scaling
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential()                               #use sequential API
model.add(keras.layers.Flatten(input_shape=[28, 28]))           #create model
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))

model.compile(loss="sparse_categorical_crossentropy",           #compile model
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs = 10,              #train model
                    validation_data = (X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize = (8,5))             #plot accuracy & loss
plt.grid = True
plt.gca().set_ylim(0,1)
plt.show

model.evaluate(X_test, y_test)                                  #test

X_new = X_test[:3]
y_new = y_test[:3]
y_proba = model.predict(X_new)
y_pred = model.predict_classes(X_new)

model.save("fashion_mnist.h5")

'''-------------------------- TIMER ------------------------------------------------'''

stop = timeit.default_timer()                                   #time program
print('\n Execution Time: ', stop - start, 's') 

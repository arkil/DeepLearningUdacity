# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 20:21:10 2018

@author: Home
"""

import numpy as np
from keras.utils import np_utils
import tensorflow as tf
tf.python.control_flow_ops = tf

np.random.seed(42)

X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

from keras.models import Sequential
from keras.layers.core import Dense, Activation
y = np_utils.to_categorical(y)
xor = Sequential()
xor.add(Dense(32, input_dim =2))
xor.add(Activation("tanh"))
xor.add(Dense(2))
xor.add(Activation('sigmoid'))
xor.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
xor.fit(X,y,nb_epoch=50)


xor.summary()

history = xor.fit(X, y, nb_epoch=100, verbose=0)

score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

print("\nPredictions:")
print(xor.predict_proba(X))
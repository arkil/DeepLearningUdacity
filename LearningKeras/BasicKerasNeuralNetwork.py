# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:33:41 2018

@author: Home
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

model = Sequential()

model.add(Dense(32, input_dim=X.shape[1]))

model.add(Activation('softmax'))
model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])
model.summary()
model.fit(X, y, epochs=1000, verbose=0)
model.evaluate()
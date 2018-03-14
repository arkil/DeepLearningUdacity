# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 00:00:27 2018

@author: Home
"""

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expl = np.exp(L)
    sumE = sum(expl)
    result =[]
    for i in expl:
        result.append(i*1.0/sumE)
    return result
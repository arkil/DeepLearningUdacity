# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 00:28:04 2018

@author: Home
"""

import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

w = np.array([0.5, -0.5, 0.3, 0.1])



h = np.dot(x,w)


nn_output = sigmoid(h)


error = y - nn_output

error_term = error * sigmoid_prime(h)


del_w = learnrate * error_term * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
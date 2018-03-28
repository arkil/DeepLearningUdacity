# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 01:31:04 2018

@author: Home
"""

# Solution is available in the other "quiz_solution.py" tab
import tensorflow as tf

def get_weights(n_features, n_labels):
    return tf.Variable(tf.truncated_normal((n_features,n_labels)))


def get_biases(n_labels):
    
    return tf.Variable(tf.zeros(n_labels))



def linear(input, w, b):
    return tf.add(tf.matmul(input,w),b)
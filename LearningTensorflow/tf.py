# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 00:26:04 2018

@author: Home
"""

import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
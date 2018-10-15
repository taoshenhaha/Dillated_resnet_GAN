# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:13:55 2018

@author: dell
"""

import tensorflow as tf

    
    
    
a = tf.constant([1,2,3.])
b=tf.constant([1,6,4.])
c=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(a,b) )

with tf.Session() as ss:
    print(ss.run([c]) )
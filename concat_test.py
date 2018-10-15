# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 18:40:52 2018

@author: dell
"""

import tensorflow  as tf
a=tf.constant([[1,2,3,4]])
b=[]

    
with tf.Session() as sess:
    for index,i in enumerate(tf.split(1,4,a)):
        b.append(i)
        c=tf.concat(1,b)
    print(sess.run(c) )
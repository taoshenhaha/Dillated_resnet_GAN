# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 00:02:39 2018

@author: dell
"""
import tensorflow as tf
lists =[[[]*3]*3]
lists[0].append(0)
print(lists)

a=tf.constant(1,shape=[2,3,4])
b=tf.constant(1,shape=[2,3,5])
c=tf.concat([a,b],2)
print(c)
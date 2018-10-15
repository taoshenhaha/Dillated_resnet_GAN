# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:58:09 2018

@author: dell
"""
import tensorflow as tf
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
print(a)
c=tf.add(a,b)
print(c)

with tf.Session() as sess:
    print(sess.run(c,feed_dict={a:10,b:30}))  #把10赋给a，30赋给b
    
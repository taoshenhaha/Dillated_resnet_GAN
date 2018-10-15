# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 01:25:17 2018

@author: dell
"""

import tensorflow as tf
a=tf.constant([ [[1,2,3,4]],[[5,6,7,8]],[[9,10,11,12]] ] )

b=tf.reshape(a,[3,2,2])
bb=tf.concat(1,(b,b))
print(bb)
with tf.Session() as sess:
    c=sess.run(bb)
    print(c)
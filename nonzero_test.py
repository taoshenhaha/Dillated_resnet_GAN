# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:06:38 2018

@author: dell
"""
import numpy as np
import os
a=[[1,2,0],
   [1,2,3],
   [1,0,3]


]

ss=zip(np.nonzero(a)[1],np.nonzero(a)[0] ) 
print(list(ss) )
x = [1, 2, 3]
y = [4, 5, 6, 7]
xy = zip(x, y)
print (list(xy) )
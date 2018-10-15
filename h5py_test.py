# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 19:57:39 2018

@author: dell
"""

import numpy as np
k = np.ones((10,10))

#np.save("a.npy", k)
c = np.load("IMG_output_1_0.npy")
print(k.dtype)
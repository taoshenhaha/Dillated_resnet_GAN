# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 15:55:32 2018

@author: dell
"""

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial import KDTree 
leafsize = 2048
vertx=np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]) 
tree=KDTree(vertx.copy(),leafsize=leafsize) 
distances, locations=tree.query(vertx,2) 
print(distances,locations)
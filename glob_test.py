# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:50:04 2018

@author: dell
"""
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import scipy
import glob
import os
root = 'ShanghaiTech_Crowd_Counting_Dataset'
print("START!!!")
#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final\train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_A='ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images'
aa=1
a=glob.glob(os.path.join(path_A, 'IMG_1.jpg'))
print("a",a)
lena = np.array(ndimage.imread("ShanghaiTech_Crowd_Counting_Dataset\\part_A_final\\train_data\\images\\IMG_1.jpg",flatten=False) )# 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
print(lena.shape) #(512, 512, 3)
print(lena)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
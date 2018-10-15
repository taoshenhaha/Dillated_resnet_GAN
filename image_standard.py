import tensorflow as tf
from PIL import Image
import random
import pickle as p
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from generate_density import *

#读取图像数据


for countss in range(Train_image):
    countss = countss + 1
    print("进度条为：", countss, "/", Train_image)
    image = np.array(Image.open(
        'ShanghaiTech_Crowd_Counting_Dataset/part_A_final/'
        'train_data/images/IMG_%s.jpg' % countss) )
    print(image)


    image[:, :, 0] = image[:, :, 0] - 108
    image[:, :, 1] = image[:, :, 1] - 95
    image[:, :, 2] = image[:, :, 2] - 92


    plt.figure(1)
    plt.imshow(image)
    plt.show()



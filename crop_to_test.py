# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:37:36 2018

@author: dell
"""

from PIL import Image
import random
import pickle as p
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from generate_density import *


def load_shanghai_batch(filename):
    with open(filename,'rb') as f:
        datadict = np.load(f, encoding='latin1')
        return datadict
#这里进行一些解释，作为选择文件的数量怎么确定还不知道呢
def main():
    for countss in range(Train_image):
        countss=countss+1
        print("进度条为：",countss,"/",Train_image)
        image = Image.open(
                           'output//train_image//IMG_%s.jpg'%countss)

        # scipy.misc.imsave('D://Deeplearning_Demo//IMG_3_crop%s.jpg'%i, reimg1)
        # 图片的宽度和高度
        img_size = image.size
        print(img_size[0])
        img_w = img_size[0]
        img_h = img_size[1]
        # 截取图片中一块宽和高都是240的
        w = Crop_W
        h = Crop_H
        count = 1
        counts = 11
        print("图片宽度是", img_w)
        print("图片高度是", img_h)

        groundtruth_x=np.array(load_shanghai_batch("output//train_groundtruth//IMG_output_%s.npy"%countss))
        #groundtruth_x=np.transpose(groundtruth_x)
        groundtruth_x=groundtruth_x*Zoom_size
        #print("npy_x=",groundtruth_x[:,1] )

        for i in range(1000):
            x=int(random.uniform(1,img_w) )
            y=int(random.uniform(1,img_h) )

            if(x+w<img_w and y+h<img_h):
                #这里进行图片的裁剪
                #print(x,y,w,h)
                #print("            ")
                region = image.crop((x, y, x+w, y+h))
                region.save("output//crop_image//IMG_%s_%s.jpg"%(countss,count) )
                #在此处加入groundtruth的裁剪
                #print(x,y,w,h)
                groundtruth_k=groundtruth_x[y:y+h,x:x+w]
                output_path="output//crop_groundtruth//IMG_output_%s_%s.npy"%(countss,count)
                np.save(output_path, groundtruth_k)

            else:
                continue

            count=count+1
            if(count==counts):
                break

        print("got done!!")

    Q_x=np.array(load_shanghai_batch("output//crop_groundtruth//IMG_output_1_1.npy"))
    #print("Q_x:",Q_x[100,100])
    i0=Image.fromarray(Q_x)
    #result=i0.show()


if __name__=="__main__":
    main()

    





    





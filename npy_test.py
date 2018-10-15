# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 11:30:53 2018

@author: dell
"""
import pickle as p
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image

#import  torchvision.transforms as transforms
import tensorflow as tf



def load_shanghai_batch(filename):
    with open(filename,'rb') as f:
        datadict = np.load(f, encoding='latin1')
        return datadict


if __name__ =="__main__":

    x=np.array(load_shanghai_batch("output//result_groundtruth//Result_50000.npy"))
    #x=np.array(load_shanghai_batch("output//train_groundtruth//IMG_output_2.npy"))
    #x=np.array(load_shanghai_batch("output//crop_groundtruth//IMG_output_16_3.npy") )
    #x=np.array(load_shanghai_batch("label//IMG_output_1_5.npy" ) )
    #x=np.array(load_shanghai_batch("label//IMG_output_3_4.npy" ) )
    #x = np.array(load_shanghai_batch("//home/yf302/mscnn-master//Data_original//Data_gt//train_gt//train_data_gt_A_4.npy"))

    #x=np.transpose(x)
    with tf.Session() as sess:
        x = tf.squeeze(x).eval()


    x = x*10000

    img0 = x[0]
    img1 = x[1]


    print("结果：", img0, img1)
    print("前端为：", x[0:10, 0:10])
    max_x=np.max(x)
    print("最大的数字为：",max_x)
    #np.where(np.max(a))


    #print("x=",x[600,0:500] )

    print(x.shape)
    x = x
    print("正在保存图片！")
    i0 = Image.fromarray(x)
    result = i0.show()




    
        
    
        
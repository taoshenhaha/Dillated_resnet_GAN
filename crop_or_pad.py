# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:45:33 2018

@author: taoshen
"""

import tensorflow as tf
import scipy.misc  

#读取图像可任意大小  
filenames = ['D://Deeplearning_Demo//IMG_3.jpg']   
filename_queue = tf.train.string_input_producer(filenames)  
reader = tf.WholeFileReader()  
key,value = reader.read(filename_queue)  
images = tf.image.decode_jpeg(value)#tf.image.decode_png(value)
aa=images.get_shape().as_list()
CP_H = 360
CP_W = 300
# 裁切图片
with tf.Session() as sess:
    
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord)  

    reshapeimg = tf.image.resize_image_with_crop_or_pad(images,CP_H,CP_W)
    #reimg1的类型是<class 'numpy.ndarray'>  
    reimg1 = reshapeimg.eval()  
    print(print(sess.run(aa)))
    scipy.misc.imsave('D://Deeplearning_Demo//IMG_3_crop.jpg', reimg1)
    
    coord.request_stop()  
    coord.join(threads)    
    print('crop_or_pad successful!')
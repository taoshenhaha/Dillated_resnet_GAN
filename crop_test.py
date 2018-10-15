# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:27:12 2018

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:45:33 2018

@author: taoshen
"""
 

import tensorflow as tf
import matplotlib.pyplot as plt
import math

image_raw_data_jpg = tf.gfile.FastGFile('D://Deeplearning_Demo//IMG_3.jpg', 'r').read()
 
with tf.Session() as sess:
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
    print(img_data_jpg.eval().shape[0]/8 )
    img_w=math.ceil(img_data_jpg.eval().shape[0]/8 )*8
    print(img_w)
    img_h=math.ceil(img_data_jpg.eval().shape[1]/8 )*8
    crop = tf.image.resize_image_with_crop_or_pad(img_data_jpg,img_w, img_h)
    pad = tf.image.resize_image_with_crop_or_pad(img_data_jpg, 2000, 2000)
    plt.imshow(crop.eval())
    plt.figure(2)
    plt.imshow(pad.eval())
    plt.figure(1)
    plt.show()
	
	 
	
    
   
     
   
	
    
	
 
	
	
   
	
	
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 20:41:16 2018

@author: dell
"""

import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

NUM_THREADS=1
batch_size=2
capacity=1000
image_W=240
image_H=240
image_D=1
image_bytes = image_H * image_W * image_D
#队列的形式读取图片和相应的密度图
#对于队列读取文件


Image_list=[]
Label_list=[]

#进行文件读取的过程！
#这里对应的结果
Image_File_dir="image"
Label_File_dir="label"

#得到了一些相应的文件名称，便于下一次调用
def get_files(file_dir,label_dir):
    for file in os.listdir(file_dir):
        Image_list.append(file_dir+'/'+ file) 
    print(Image_list)
    for file in os.listdir(label_dir):
        Label_list.append(label_dir+'/'+ file) 

def get_batch(image_path, label_path, image_W, image_H, batch_size, capacity,sess=None):
    #转换类型
    image_path = tf.cast(image_path, tf.string)
    label_path = tf.cast(label_path, tf.string)
    
   
        
    
    # make an input queue
    input_queue = tf.train.slice_input_producer([image_path, label_path])
    """
    利用read_file得到了image的信息
    
    """
    
    image_contents = tf.read_file(input_queue[0]) #read img from a queue  
    #image_reader = tf.WholeFileReader()  #reader
    #_, image_contents = image_reader.read(input_queue[0])  #reader读取序列
    print("结束")
#step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3) 
    print(image.get_shape() )
    
#step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    
    image = tf.image.per_image_standardization(image)
    
 
    
    
    
    """
    label = input_queue[1]
    利用read_file得到了image的信息
    
    """
    print("Done")
    
    label=tf.read_file(input_queue[1])  #reader读取序列
 
    # 读出的 value 是 string，现在转换为 uint8 型的向量
    record_bytes = tf.decode_raw(label, tf.float32)

    depth_major = tf.reshape(tf.slice(record_bytes, [20], [240*240*1]),                       
                         [1, 240, 240])#depth, height, width
    print("done")
    uint8image = tf.transpose(depth_major, [1, 2, 0])
    
    label = tf.cast(uint8image, tf.float32)
    
#step4：生成batch
#image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32 
#label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= NUM_THREADS, 
                                                capacity = capacity)
    #重新排列label，行数为[batch_size]
    #label_batch = tf.reshape(label_batch, [batch_size])
    
    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.cast(label_batch, tf.float32)
    return image_batch, label_batch 
  
def get_batch_image(image_path, image_W, image_H, batch_size, capacity,sess=None):
    #转换类型
    image_path = tf.cast(image_path, tf.string)
    
    
   
        
    
    # make an input queue
    print("Start")
    input_queue = tf.train.slice_input_producer([image_path])
    
    
    """
    利用read_file得到了image的信息
    
    """
    image_reader = tf.WholeFileReader() #read img from a queue
    
    _, image = image_reader.read(input_queue)
    print("Done")
    image = tf.image.decode_jpeg(image)
    
       
     
     
    
#step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    #image = tf.image.decode_jpeg(image_contents, channels=3) 
    
#step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    
    image = tf.image.per_image_standardization(image)
    
 
    
    
    
    """
    label = input_queue[1]
    利用read_file得到了image的信息
    
    """
   
#step4：生成batch
#image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32 
#label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch= tf.train.batch(image,
                                                batch_size= batch_size,
                                                num_threads= NUM_THREADS, 
                                                capacity = capacity)
    #重新排列label，行数为[batch_size]
    #label_batch = tf.reshape(label_batch, [batch_size])
    
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch

"""




首先是进行转换读取文件的名称，把他们转换成list的形式例如[image_list,label_list]



"""

"""
接下来进行试验


"""
#调用get_files函数，进行Image_File_dir,Label_File_dir的填充
get_files(Image_File_dir,Label_File_dir)

Image_batch, Label_batch=get_batch(Image_list, Label_list,image_W, image_H, batch_size, capacity)
#Image_batch, Label_batch=get_batch_image(Image_list,image_W, image_H, batch_size, capacity)
with tf.Session() as sess:
    coord = tf.train.Coordinator() #协同启动的线程  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) #启动线程运行队列 
    
    for i in range(10):
        e_val, l_val = sess.run([Image_batch, Label_batch])
        print (e_val, l_val)
        """
        sess.run(image)
        print(type(image))#tensor
        print(type(image.eval()))#ndarray
        print(image.eval().shape)#240×320×3
        print(image.eval().dtype)#uint8
        plt.figure(1)
        plt.imshow(image.eval()) 
        plt.show()
        """
        
    
    
    
    coord.request_stop() #停止所有的线程  
    coord.join(threads)
    
    


#print(Image_batch.shape())




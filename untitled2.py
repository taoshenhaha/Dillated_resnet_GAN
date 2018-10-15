# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 15:05:14 2018

@author: dell
"""



import matplotlib.pyplot as plt
import tensorflow as tf     
import numpy as np
tf.reset_default_graph()     
image_path = r'D:/Deeplearning_Demo/image/IMG_1_0.jpg'#绝对路径
label_path=r'D:/Deeplearning_Demo/image/IMG_output_1_0.npy'
xia_path="ddddd"

image_path = tf.convert_to_tensor(image_path)
label_path = tf.convert_to_tensor(label_path)

print("strat")
file_queue = tf.train.string_input_producer([label_path]) #创建输入队列
#file_queue = tf.train.slice_input_producer([[image_path],[label_path] ])
#image = tf.train.slice_input_producer([[image_path] ])
#file_queue=tf.convert_to_tensor(file_queue)
#file_queue[0]= tf.convert_to_tensor(file_queue[0])

#reader = tf.FixedLengthRecordReader(record_bytes=240*240*1)
#key,image = reader.read(file_queue)


"""
标签数据

"""

image=tf.read_file(file_queue)  #reader读取序列
 
# 读出的 value 是 string，现在转换为 uint8 型的向量
record_bytes = tf.decode_raw(image, tf.float32)

depth_major = tf.reshape(record_bytes,                       
                        [1, 240, 240])#depth, height, width
print("done")
uint8image = tf.transpose(depth_major, [1, 2, 0])
    
label = tf.cast(uint8image, tf.float32)
print(label)

with tf.Session() as sess: 
    sess.run(tf.local_variables_initializer() )
    sess.run(tf.global_variables_initializer() ) 
    coord = tf.train.Coordinator() #协同启动的线程  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) #启动线程运行队列  
    sess.run(image)
    #print(label.eval() )
    print(type(image))#tensor
    coord.request_stop() #停止所有的线程  
    coord.join(threads)
    print(type(image.eval()))#ndarray
    print(image.eval().shape)#240×320×3
    print(image.eval().dtype)#uint8
    plt.figure(1)
    plt.imshow(image.eval()) 
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 21:39:20 2018

@author: dell
"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from dilated_resnet import *

tf.reset_default_graph()
# image_path = r'D:/Deeplearning_Demo/image/IMG_1_0.jpg'#绝对路径
# label_path=r'IMG_output_1_0.npy'
# xia_path="ddddd"
"""
Image_File_dir = "output/crop_image"
Label_File_dir = "output/crop_groundtruth"

"""
Image_File_dir = "image"
Label_File_dir = "label"
Image_list = []
Label_list = []
learning_rate = 0.0001
# batch_size=1
each_step = 10

# 定义存储路径
ckpt_dir = "./model"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# 得到了一些相应的文件名称，便于下一次调用
def get_files(file_dir, label_dir):
    for file in os.listdir(file_dir):
        Image_list.append(file_dir + '/' + file)
        Image_list.sort()

    print(Image_list)

    for file in os.listdir(label_dir):
        Label_list.append(label_dir + '/' + file)
        Label_list.sort()


    print(Label_list)



# 损失函数的计算方法是什么，这里我们进行相应的改进比如可以改进成一些方差的计算方法等等
def loss(y, pred_y):
    loss = tf.reduce_sum(tf.square(y - pred_y))
    return loss


get_files(Image_File_dir, Label_File_dir)
image_path = tf.convert_to_tensor(Image_list)
label_path = tf.convert_to_tensor(Label_list)

print("strat")
# file_queue = tf.train.string_input_producer([image_path]) #创建输入队列
file_queue = tf.train.slice_input_producer([image_path, label_path], shuffle=True)
# image = tf.train.slice_input_producer([[image_path] ])
file_queue = tf.convert_to_tensor(file_queue)
# file_queue[0]= tf.convert_to_tensor(file_queue[0])

# reader = tf.WholeFileReader()
# key,image = reader.read(file_queue)

"""
图像数据
"""
image = tf.read_file(file_queue[0])  # reader读取序列
image = tf.image.decode_jpeg(image, channels=3)  # 解码，tensor
image = tf.image.resize_images(image, [240, 240])
"""
标签数据

"""

label = tf.read_file(file_queue[1])  # reader读取序列

# 读出的 value 是 string，现在转换为 uint8 型的向量
record_bytes = tf.decode_raw(label, tf.float32)

depth_major = tf.reshape(tf.slice(record_bytes, [32], [240 * 240 * 1]),
                         [1, 240, 240])  # depth, height, width
print("done")
uint8image = tf.transpose(depth_major, [1, 2, 0])

label = tf.cast(uint8image, tf.float32)

"""这里需要用参数进行设计"""
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=8,
                                          num_threads=1,
                                          capacity=1)



# saving and loading networks
# 保存模型的参数等等


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  # 协同启动的线程

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动线程运行队列

    # 保存模型





    # 这里对应着global_step的大小
    
    # 下面开始循环的过程
    for i in range(1000):
        # 应用到的网络结构
        print("i的数字为:", i)
        print(record_bytes.eval().size)
        if i%10==0:
            for j in range(8):
                print("J:",j)


                # losses=sess.run(loss_result)
                [image, label] = sess.run([image_batch, label_batch])
                print(image.shape)
                print(label.shape)
                # 显示原始数据的分布是什么
                print_label=tf.squeeze(label[j, :, :, :]).eval()
                print_label=print_label*10000
                i0 = Image.fromarray(print_label)
                #plt.figure(i+1)
                result = i0.show()
                print("jieguo:", label[:, 1:2, 1:2, :])
                print(type(image))  # tensor
                image = tf.squeeze(image[j, :, :, :]).eval()
                print("image_type", type(image))  # ndarray
                print(image.shape)  # 240×320×3
                image = tf.cast(image, tf.uint8).eval()
                print("image.dtype:", image.dtype)  # uint8

                plt.figure(1)
                plt.imshow(image)
                plt.show()
                print("进行第二次测试的过程：")



                [image, label] = sess.run([image_batch, label_batch])

                print(image.shape)
                print(label.shape)
                # 显示原始数据的分布是什么
                print_label = tf.squeeze(label[j, :, :, :]).eval()
                print_label = print_label * 10000
                i0 = Image.fromarray(print_label)
                # plt.figure(i+1)
                result = i0.show()
                print("jieguo:", label[:, 1:2, 1:2, :])
                print(type(image))  # tensor
                image = tf.squeeze(image[j, :, :, :]).eval()
                print("image_type", type(image))  # ndarray
                print(image.shape)  # 240×320×3
                image = tf.cast(image, tf.uint8).eval()
                print("image.dtype:", image.dtype)  # uint8

                plt.figure(1)
                plt.imshow(image)
                plt.show()
        

    # 停止所有的线程
    coord.request_stop()
    coord.join(threads)

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
from generate_density import Zoom_size
tf.reset_default_graph()
# image_path = r'D:/Deeplearning_Demo/image/IMG_1_0.jpg'#绝对路径
# label_path=r'IMG_output_1_0.npy'
# xia_path="ddddd"

"""
Image_File_dir = "image"
Label_File_dir = "label"
"""

Image_File_dir = "output/crop_image"
Label_File_dir = "output/crop_groundtruth"



"""
Image_File_dir = "D:/Deeplearning_Demo/TensorFlow_demo/Ours_crowd_counting/output/crop_image"
Label_File_dir = "D:/Deeplearning_Demo/TensorFlow_demo/Ours_crowd_counting/output/crop_groundtruth"
"""
Image_list = []
Label_list = []
learning_rate_base = 0.1

learning_decay_steps = 1500

learning_rate_decay = 0.96
# 这里为什么会出问题?
# batch_size=2
each_step = 10000
#Zoom_size = 1

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
    #loss=tf.reduce_sum(tf.square(y-pred_y) )
    y_num=tf.reduce_sum(y)
    #y_num=np.sum(y)
    #pre_y_num=np.sum(pred_y)
    pre_y_num=tf.reduce_sum(pred_y)
    #+0.0001*tf.reduce_sum(tf.square((y_num-pre_y_num)) )
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_y, labels=y))\
                          +0.0001*tf.reduce_sum(tf.square((y_num-pre_y_num)) )



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
#image = tf.image.per_image_standardization(image)
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

label = tf.cast(uint8image, tf.float32) / (Zoom_size)

"""这里需要用参数进行设计"""
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=batch_size,
                                          num_threads=1,
                                          capacity=10000)

global_step = tf.Variable(0, trainable=False)  # 给轮数计数器赋初值，并将其设置为不可训练

# 定义指数衰减学习率,梯形的学习率或者转换成每一步都需要进行相应的改变的学习率
learning_rate = tf.train.exponential_decay(
    learning_rate_base,
    global_step,
    learning_decay_steps,
    learning_rate_decay,
    staircase=True)


"""
训练的程序
"""
predict_map, D_1, D_2 = inference(image_batch)
print("验证的结果：", predict_map, label_batch)
# (1e+13)或者10000,* ((1e+13))
total_loss = loss(label_batch,predict_map ) / (batch_size)
# GradientDescentOptimizer AdamOptimizer RMSPropOptimizer
loss_result = tf.train.RMSPropOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
# for tv in tf.trainable_variables():
# print("变量名称：", tv.name)
# w = tf.get_default_graph().get_tensor_by_name("DRN_1/weight/DRN_1_va:0")


# saving and loading networks
# 保存模型的参数等等
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 占用GPU40%的显存
#session = tf.Session()
with tf.Session(config=config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  # 协同启动的线程

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动线程运行队列

    # 保存模型
    # 这里对应着global_step的大小
    step = 0
    # 现在开始加载一些预先训练好的模型参数

    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # 下面开始循环的过程
    for i in range(300000):
        # 应用到的网络结构
        print("轮数是:", i)
        global_step = i
        # print("第一层的权重可视化：", sess.run(w))


        # 进行相应的训练过程
        losses = sess.run(loss_result)




        if i % each_step == 0:
            loss_print = sess.run(total_loss)

            print("损失结果为：", loss_print)

        if i%learning_decay_steps == 0:
            print("学习率为:", sess.run(learning_rate))


            # saver.save(sess, ckpt_dir+"/my-model.ckpt", global_step=i)
        if i % each_step == 0:
            # 接下来显示训练一段时间的测试结果如何，来判断这种方法是否应该用的
            [image, label] = sess.run([image_batch, label_batch])
            # print(image.shape)
            print(label.shape)
            #label[0:1, 1:5, 1:5, :]
            print("标签为:", label[0, 100:220, 100:220, :])

            print(type(image))  # tensor

            # 转换了图片的类型
            image = tf.squeeze(image[0, :, :, :]).eval()
            print("image_type", type(image))  # ndarray
            # print("image:",image)
            print(image.shape)  # 240×320×3
            image = tf.cast(image, tf.uint8).eval()
            print("image.dtype:", image.dtype)  # uint8



            print("变化之前的label：",label.shape)

            # 现在要转换原始的label以及生成的结果
            label = tf.squeeze(label[0, :, :, :]).eval()
            print("label_type", type(label))  # ndarray
            print(label.shape)  # 240×320×3
            label = tf.cast(label, tf.float32).eval()
            # print("label.dtype:", label.dtype)  # uint8

            # 开始转换一下生成结果
            print_label=label*10000

            # 显示原始数据的分布是什么
            i0 = Image.fromarray(print_label)
            #plt.figure(i+1)
            result = i0.show()

            # 显示一下训练出来的数据是什么样子的
            result_groundtruth, DD_1, DD_2 = sess.run([predict_map, D_1, D_2])


            print(result_groundtruth.shape )

            result_groundtruth_1 = tf.squeeze(result_groundtruth[0, 100:220, 100:220, :]).eval()
            result_groundtruth_2 = tf.squeeze(result_groundtruth[0, 1, 1, :]).eval()

            result_groundtruth = tf.squeeze(result_groundtruth[0, : , :,:]).eval()
            # 得到的第一层的结果
            DD_1 = tf.squeeze(DD_1[0, 1:5, 1:5, :]).eval()





            print("不加入激活层是：", DD_1)
            print("得到的结果是：", result_groundtruth)
            print("结果中间部分是：",result_groundtruth_1)
            print(" 角落部分是：", result_groundtruth_2)
            # print("得到的结果是：", result_groundtruth.dtype)
            # 为什么得到的数据会那么的大呢？
            print_result_groundtruth=result_groundtruth*10000

            i1 = Image.fromarray(print_result_groundtruth)
            #plt.figure(i+2)
            result = i1.show()
            result_groundtruth=np.array(result_groundtruth)
            result_save_path="output/result_groundtruth/Result_%d.npy" % i
            np.save(result_save_path,result_groundtruth)


            saver.save(sess, ckpt_dir + "/my-model.ckpt", global_step=i)
            #显示图片
            plt.figure(i)
            plt.imshow(image)
            #plt.show()

    # 停止所有的线程
    coord.request_stop()
    coord.join(threads)


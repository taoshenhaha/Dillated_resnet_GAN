# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:26:44 2018

@author: dell
"""
import tensorflow as tf
import numpy as np
from PIL import Image
"""
im = Image.open('D://Deeplearning_Demo//Data_original//Data_im//train_im//IMG_1_B.jpg')
im.show()
im_array = np.array(im)
print(im_array.shape)
"""
from PIL import Image
from dilated_resnet import *
import random
import pickle as p
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
label_path='IMG_output_1_0.npy'
#label_path=tf.convert_to_tensor(label_path)
#定义定义的参数的存储位置在cpu还是在GPu上 tf.contrib.layers.xavier_initializer()
def create_variables(name,shape,initializer=tf.truncated_normal_initializer(),is_fc_layers=False):

    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
    #new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
    #                               regularizer=regularizer)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer
                                   )
    return new_variables





# 这里先构建一个批归一化层的函数，方便后续使用,训练和测试的时候需要改变一下把？
def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, varience = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, varience, beta, gamma, BN_EPSILON)
    return bn_layer


# 在这里进行相应的修改操作吧？
def conv_bn_relu_layer(input_layer, filter_shape, stride, dilation=1):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)
    if dilation == 1:
        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    elif dilation == 2:
        conv_layer = tf.nn.atrous_conv2d(value=input_layer, filters=filter, rate=2, padding='SAME')
    elif dilation == 4:
        conv_layer = tf.nn.atrous_conv2d(value=input_layer, filters=filter, rate=4, padding='SAME')
    else:
        raise ValueError('no dilation is choosen!!!')

    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


# 我们采用这个基础网络结构
def bn_relu_conv_layer(input_layer, filter_shape, stride=1, dilation=1):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    # 这里进行卷积操作的一个很重要的部分，选择卷积的方式是什么，其中有1,2,4
    if dilation == 1:
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    elif dilation == 2:
        conv_layer = tf.nn.atrous_conv2d(value=relu_layer, filters=filter, rate=2, padding='SAME')
    elif dilation == 4:
        conv_layer = tf.nn.atrous_conv2d(value=relu_layer, filters=filter, rate=4, padding='SAME')
    else:
        raise ValueError('no dilation is choosen!!!')
    return conv_layer


# 开始构造block的结构过程
def residual_block(input_layer, output_channel, dilation=1, stride=1, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # 按照正常的网络结构
    # 我们选择先bath 然后进行卷积和relu的操作
    # Y = conv(Relu(batch_normalize(X)))
    # bn_relu_conv_layer(input_layer,filter_shape,stride,dilation=1)
    '''
    设置残差网络的模板的第一层
    '''
    # 这里进行查看输入输出维度以及深度是否相同,这里没考虑网络卷积之后的大小是否跟原始大小一样的！




    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride, dilation)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], stride, dilation)

    # mid_layer=bn_relu_conv_layer(input_layer,[3,3],stride,dilation=dilation)

    # out_layer=bn_relu_conv_layer(mid_layer,[3,3],stride,dilation=dilation)



    if input_channel != output_channel:
        filter = create_variables(name='Unichannel', shape=[1, 1, input_channel, output_channel])
        input_layer = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')

    result = tf.add(input_layer, conv2)
    return result




def inference(input_layer):
    #下面按照那个结构进行相应的搭建框架的结构如下所示：[batch,h,w,3]出去的时候也应该是[batch,h,w,3]
    """
    这下面的数字以后会改的
    """
    input_channel=input_layer.get_shape().as_list()[-1]
    input_w=input_layer.get_shape().as_list()[2]
    input_h=input_layer.get_shape().as_list()[1]
    #[7,7,16]
    with tf.variable_scope('DRN_1'):
        filter=create_variables(name='DRN_1_va',shape=[7,7,input_channel,16])
        DRN_1=tf.nn.conv2d(input_layer,filter,strides=[1,1,1,1],padding='SAME')
        DRN_1=tf.nn.relu(DRN_1)

    with tf.variable_scope('DRN_2'):
        DRN_2=residual_block(DRN_1,16,dilation=1,stride=1,first_block=False)

        # [3,3,32]+[3,3,32]的残差网络的设,stride设计为2相当于进行了降采样的操作
    with tf.variable_scope('DRN_3'):
        DRN_3 = residual_block(DRN_2, 32, dilation=1, stride=1, first_block=False)
    with tf.variable_scope('DRN_4_2'):
        DRN_4 = residual_block(DRN_3, 64, dilation=1, stride=1, first_block=False)
    with tf.variable_scope('DRN_5_1'):
        DRN_5=residual_block(DRN_4,128,dilation=1,stride=1,first_block=False)
    with tf.variable_scope('DRN_5_2'):
        DRN_5=residual_block(DRN_5,128,dilation=1,stride=1,first_block=False)

    # [3,3,256]+[3,3,256]的残差网络的设,stride设计为2相当于进行了降采样的操作
    with tf.variable_scope('DRN_6_1'):
        DRN_6 = residual_block(DRN_5, 256, dilation=2, stride=1, first_block=False)
    with tf.variable_scope('DRN_6_2'):
        DRN_6 = residual_block(DRN_5, 256, dilation=2, stride=1, first_block=False)
        #DRN_6=tf.nn.sigmoid(DRN_6)
    return DRN_6



# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 21:39:20 2018

@author: dell
"""



tf.reset_default_graph()
# image_path = r'D:/Deeplearning_Demo/image/IMG_1_0.jpg'#绝对路径
# label_path=r'IMG_output_1_0.npy'
# xia_path="ddddd"

Image_File_dir = "D:/Deeplearning_Demo/image"
Label_File_dir = "D:/Deeplearning_Demo/label"

"""
Image_File_dir = "D:/Deeplearning_Demo/TensorFlow_demo/Ours_crowd_counting/output/crop_image"
Label_File_dir = "D:/Deeplearning_Demo/TensorFlow_demo/Ours_crowd_counting/output/crop_groundtruth"
"""
Image_list = []
Label_list = []
learning_rate = 0.01
# 这里为什么会出问题?
# batch_size=2
each_step = 10

# 定义存储路径
ckpt_dir = "./model"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# 得到了一些相应的文件名称，便于下一次调用
def get_files(file_dir, label_dir):
    for file in os.listdir(file_dir):
        Image_list.append(file_dir + '/' + file)
    print(Image_list)
    for file in os.listdir(label_dir):
        Label_list.append(label_dir + '/' + file)


# 损失函数的计算方法是什么，这里我们进行相应的改进比如可以改进成一些方差的计算方法等等
def loss(y, pred_y):
    loss = tf.reduce_sum(tf.square(y - pred_y))
    return loss


get_files(Image_File_dir, Label_File_dir)
image_path = tf.convert_to_tensor(Image_list)
label_path = tf.convert_to_tensor(Label_list)

print("start")
# file_queue = tf.train.string_input_producer([image_path]) #创建输入队列
file_queue = tf.train.slice_input_producer([image_path, label_path], shuffle=False)
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
image = tf.image.per_image_standardization(image)
"""
标签数据

"""

label = tf.read_file(file_queue[1])  # reader读取序列

# 读出的 value 是 string，现在转换为 uint8 型的向量
record_bytes = tf.decode_raw(label, tf.float32)

depth_major = tf.reshape(tf.slice(record_bytes, [20], [240 * 240 * 1]),
                         [1, 240, 240])  # depth, height, width
print("done")
uint8image = tf.transpose(depth_major, [1, 2, 0])

label = tf.cast(uint8image, tf.float32)/30000

"""这里需要用参数进行设计"""
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=batch_size,
                                          num_threads=1,
                                          capacity=10000)

predict_map= inference(image_batch)

print("验证的结果：", predict_map, label_batch)

# saving and loading networks
# 保存模型的参数等等


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  # 协同启动的线程

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动线程运行队列

    # 保存模型







    # 下面开始循环的过程
    for i in range(100):
        # 应用到的网络结构
        print("i的数字为:", i)

        #losses = sess.run(loss_result)




            # saver.save(sess, ckpt_dir+"/my-model.ckpt", global_step=i)

        # 接下来显示训练一段时间的测试结果如何，来判断这种方法是否应该用的
        [image, label] = sess.run([image_batch, label_batch])
        print(image.shape)
        print(label.shape)
        print("jieguo:", label[:, 1:10, 1:10, :])
        print(type(image))  # tensor

        # 转换了图片的类型
        image = tf.squeeze(image[0:1, :, :, :]).eval()
        print("image_type", type(image))  # ndarray
        print("image:", image)
        print(image.shape)  # 240×320×3
        image = tf.cast(image, tf.uint8).eval()
        print("image.dtype:", image.dtype)  # uint8

        plt.figure(i)
        plt.imshow(image)
        plt.show()

        # 现在要转换原始的label以及生成的结果
        label = tf.squeeze(label[0:1, :, :, :]).eval()
        print("label_type", type(label))  # ndarray
        print(label.shape)  # 240×320×3
        # label = tf.cast(label, tf.float32).eval()
        # print("label.dtype:", label.dtype)  # uint8



        # 开始转换一下生成结果








        # 显示原始数据的分布是什么
        i0 = Image.fromarray(label)
        # plt.figure(i+1)
        result = i0.show()

        # 显示一下训练出来的数据是什么样子的
        result_groundtruth= sess.run(predict_map)
        print("ddd:",result_groundtruth)
        result_groundtruth = tf.squeeze(result_groundtruth).eval()
        print("形状为：", result_groundtruth.shape)
        print("生成的结果：",result_groundtruth)

    # 停止所有的线程
    coord.request_stop()
    coord.join(threads)



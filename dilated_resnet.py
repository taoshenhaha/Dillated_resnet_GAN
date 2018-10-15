'''
@Function: Structure of my crowd counting


@Author: zhao houtao
@Code verification: zhao houtao
@说明：
    学习率：1e-4
    平均loss : 14.
@Data: Jul. 21, 2018
@Version: 0.1
'''

# 系统模块
import re
# 深度学习的模型库
import tensorflow as tf
# 载入项目模块
from PIL import Image
import random
import pickle as p
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
import dilated_resnet_train as DRN
import numpy as np
from cnn_basenet import *
# from hyper_parameters import *
# 模型参数设置
MP_NAME = 'mp'
train_log = 'train_log'
model = 'model'
output = 'output'
data_train_gt = 'Data_original/Data_gt/train_gt/'
data_train_im = 'Data_original/Data_im/train_im/'
data_train_index = 'Data_original/dir_name.txt'
batch_size = 4

RGB_R=tf.constant(108,name='RGB_R')
RGB_G=tf.constant(95,name='RGB_G')
RGB_B=tf.constant(92,name='RGB_B')
"""
使用flags定义命令行参数



"""
BN_EPSILON = 0.001


def _activation_summary(x):
    """
    概要汇总函数
    :param x: 待保存变量
    :return: None
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    # 记录x中0的个数，不知道是为什么这样做？
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


class BatchNorm(object):
    """
    BN操作类
    """

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        """
        初始化函数
        :param epsilon: 精度
        :param momentum: 动量因子
        :param name: name_scope
        """
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x):
        """
        BN算子
        :param x: 输入变量
        :return:
        """
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None,
                                            epsilon=self.epsilon, scale=True, scope=self.name)


def _variable_on_cpu(name, shape, initializer):
    """
    创建变量
    :param name: name_scope
    :param shape: tensor维度
    :param initializer: 初始化值
    :return: tensor变量
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)

    return var


def create_variables(name, shape, stddev=0.01, wd=0.0005):
    """
    创建有权重衰减项的变量
    :param name: name_scope
    :param shape: tensor维度
    :param stddev: 用于初始化的标准差
    :param wd: 权重
    :return: tensor变量
    """
    # wd 为衰减因子,若为None则无衰减项
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


"""
#定义定义的参数的存储位置在cpu还是在GPu上 tf.contrib.layers.xavier_initializer() tf.truncated_normal_initializer()
def create_variables(name,shape,initializer=tf.contrib.layers.xavier_initializer(),is_fc_layers=False,reuse=False):

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
    with tf.variable_scope("weight",reuse=reuse):
        new_variables = tf.get_variable(name, shape=shape, initializer=initializer
                                        )

    return new_variables


"""


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


def conv_bn_relu_layer(input_layer,filter_shape,stride,dilation=1):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''
    out_channel = filter_shape[-1]
    filter = create_variables(name='cbrl_conv', shape=filter_shape)
    if dilation==1:
        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    elif dilation==2:
        conv_layer=tf.nn.atrous_conv2d(value=input_layer, filters=filter, rate=2, padding='SAME')
    elif dilation==4:
        conv_layer=tf.nn.atrous_conv2d(value=input_layer, filters=filter, rate=4, padding='SAME')
    elif dilation == 5:
        conv_layer = tf.nn.atrous_conv2d(value=input_layer, filters=filter, rate=5, padding='SAME')
    else:
        raise ValueError('no dilation is choosen!!!')
    biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0))
    conv_layer = tf.nn.bias_add(conv_layer, biases)
    bn = BatchNorm()
    bn_layer = bn(conv_layer)
    #bn_layer = batch_normalization_layer(, out_channel)

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
    elif dilation == 5:
        conv_layer = tf.nn.atrous_conv2d(value=relu_layer, filters=filter, rate=5, padding='SAME')
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
            biases = _variable_on_cpu('biases', [output_channel], tf.constant_initializer(0))
            conv1 = tf.nn.bias_add(conv1, biases)

            bn = BatchNorm()
            conv1 = bn(conv1)
            conv1 = tf.nn.relu(conv1)
        else:
            conv1 = conv_bn_relu_layer(input_layer, [3, 3, input_channel, output_channel], stride, dilation)
            # 这里进行相应的网络拓展


            cat_conv1_shape = conv1.get_shape().as_list()[-1]
            filter = create_variables(name='catconv1_conv', shape=[1, 1, cat_conv1_shape, output_channel])
            conv1 = tf.nn.conv2d(conv1, filter, strides=[1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('conv1_biases', [output_channel], tf.constant_initializer(0))
            conv1 = tf.nn.bias_add(conv1, biases)


            with tf.variable_scope('cat_bn'):
                bn = BatchNorm()
                conv1 = bn(conv1)
                conv1 = tf.nn.relu(conv1)

    with tf.variable_scope('conv2_in_block'):
        conv2 = conv_bn_relu_layer(conv1, [3, 3, output_channel, output_channel], stride, dilation)
        #print("出现问题的维度为：",output_channel)



        cat_conv2_shape = conv2.get_shape().as_list()[-1]
        #print("cat_conv2_shape:",cat_conv2_shape)
        filter = create_variables(name='catconv2_conv', shape=[1, 1, cat_conv2_shape, output_channel])
        conv2 = tf.nn.conv2d(conv2, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('conv2_biases', [output_channel], tf.constant_initializer(0))
        conv2 = tf.nn.bias_add(conv2, biases)
        with tf.variable_scope('cat_bn'):
            bn = BatchNorm()
            conv2 = bn(conv2)
            conv2 = tf.nn.relu(conv2)

    # mid_layer=bn_relu_conv_layer(input_layer,[3,3],stride,dilation=dilation)

    # out_layer=bn_relu_conv_layer(mid_layer,[3,3],stride,dilation=dilation)

    if input_channel != output_channel:
        filter = create_variables(name='Unichannel', shape=[1, 1, input_channel, output_channel])
        input_layer = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [output_channel], tf.constant_initializer(0))
        input_layer = tf.nn.bias_add(input_layer, biases)

    result = tf.add(input_layer, conv2)
    bn = BatchNorm()
    result = bn(result)
    result = tf.nn.relu(result)

    return result

# 开始构造block的结构过程
def residual_block_1(input_layer, output_channel, dilation=1, stride=1, first_block=False):
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
            biases = variable_on_cpu('biases', [output_channel], tf.constant_initializer(0))
            conv1 = tf.nn.bias_add(conv1, biases)

            bn = BatchNorm()
            conv1 = bn(conv1)
            conv1 = tf.nn.relu(conv1)
        else:
            conv1 = conv_bn_relu_layer(input_layer, [3, 3, input_channel, output_channel], stride, dilation)
            # 这里进行相应的网络拓展




    with tf.variable_scope('conv2_in_block'):
        conv2 = conv_bn_relu_layer(conv1, [3, 3, output_channel, output_channel], stride, dilation)
        #print("出现问题的维度为：",output_channel)






    # mid_layer=bn_relu_conv_layer(input_layer,[3,3],stride,dilation=dilation)

    # out_layer=bn_relu_conv_layer(mid_layer,[3,3],stride,dilation=dilation)

    if input_channel != output_channel:
        filter = create_variables(name='Unichannel', shape=[1, 1, input_channel, output_channel])
        input_layer = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [output_channel], tf.constant_initializer(0))
        input_layer = tf.nn.bias_add(input_layer, biases)

    result = tf.add(input_layer, conv2)
    bn = BatchNorm()
    result = bn(result)
    result = tf.nn.relu(result)

    return result
def inference_multi(input_layer):
    input_channel = input_layer.get_shape().as_list()[-1]
    input_w = input_layer.get_shape().as_list()[2]
    input_h = input_layer.get_shape().as_list()[1]

    # first block
    with tf.variable_scope('D_1_1'):
        filter = create_variables(name='DRN_1_1_var', shape=[11, 11, input_channel, 16])
        D_1_1 = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
        # D_1_1 = tf.nn.atrous_conv2d(value=input_layer, filters=filter, rate=1, padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_1_1 = tf.nn.bias_add(D_1_1, biases)
        bn = BatchNorm()
        D_1_1 = bn(D_1_1)
        D_1_1 = tf.nn.relu(D_1_1)

    D_1_shape = D_1_1.get_shape().as_list()[-1]

    with tf.variable_scope('D_1_2'):
        filter = create_variables(name='DRN_1_2_var', shape=[9, 9, D_1_shape, 24])
        # D_1_2 = tf.nn.atrous_conv2d(value=D_1_1, filters=filter, rate=1, padding='SAME')
        D_1_2 = tf.nn.conv2d(D_1_1, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0))

        D_1_2 = tf.nn.bias_add(D_1_2, biases)
        bn = BatchNorm()
        D_1_2 = bn(D_1_2)
        D_1_2 = tf.nn.relu(D_1_2)

    """
    池化层第一层
    """

    with tf.variable_scope('D_1_2_pool'):
        D_1_2 = tf.nn.max_pool(D_1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
                               )

    D_2_shape = D_1_2.get_shape().as_list()[-1]

    with tf.variable_scope('D_1_3'):
        filter = create_variables(name='DRN_1_3_var', shape=[7, 7, D_2_shape, 16])
        # D_1_3 = tf.nn.atrous_conv2d(value=D_1_2, filters=filter, rate=1, padding='SAME')
        D_1_3 = tf.nn.conv2d(D_1_2, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_1_3 = tf.nn.bias_add(D_1_3, biases)
        bn = BatchNorm()
        D_1_3 = bn(D_1_3)
        D_1_3 = tf.nn.relu(D_1_3)

    D_3_shape = D_1_3.get_shape().as_list()[-1]

    with tf.variable_scope('D_1_4'):
        filter = create_variables(name='DRN_1_4_var', shape=[7, 7, D_3_shape, 16])
        # D_1_4 = tf.nn.atrous_conv2d(value=D_1_3, filters=filter, rate=1, padding='SAME')
        D_1_4 = tf.nn.conv2d(D_1_3, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_1_4 = tf.nn.bias_add(D_1_4, biases)
        bn = BatchNorm()
        D_1_4 = bn(D_1_4)
        D_1_4 = tf.nn.relu(D_1_4)

    D_4_shape = D_1_4.get_shape().as_list()[-1]

    """
    池化层第二层
    """
    with tf.variable_scope('D_1_4_pool'):
        D_1_4 = tf.nn.max_pool(D_1_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
                               )

    with tf.variable_scope('D_1_5'):
        filter = create_variables(name='DRN_1_5_var', shape=[5, 5, D_4_shape, 16], stddev=0.001)
        # D_1_5 = tf.nn.atrous_conv2d(value=D_1_4, filters=filter, rate=1, padding='SAME')
        D_1_5 = tf.nn.conv2d(D_1_4, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_1_5 = tf.nn.bias_add(D_1_5, biases)
        bn = BatchNorm()
        D_1_5 = bn(D_1_5)
        D_1_5 = tf.nn.relu(D_1_5)

    D_5_shape = D_1_5.get_shape().as_list()[-1]

    with tf.variable_scope('D_1_6'):
        filter = create_variables(name='DRN_1_6_var', shape=[5, 5, D_5_shape, 16], stddev=0.001)
        # D_1_6 = tf.nn.atrous_conv2d(value=D_1_5, filters=filter, rate=1, padding='SAME')
        D_1_6 = tf.nn.conv2d(D_1_5, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_1_6 = tf.nn.bias_add(D_1_6, biases)
        bn = BatchNorm()
        D_1_6 = bn(D_1_6)
        D_1_6 = tf.nn.relu(D_1_6)

    D_6_shape = D_1_6.get_shape().as_list()[-1]

    with tf.variable_scope('D_1_7'):
        filter = create_variables(name='DRN_1_7_var', shape=[3, 3, D_6_shape, 8], stddev=0.001)
        # D_1_7 = tf.nn.atrous_conv2d(value=D_1_6, filters=filter, rate=1, padding='SAME')
        D_1_7 = tf.nn.conv2d(D_1_6, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0))

        D_1_7 = tf.nn.bias_add(D_1_7, biases)
        bn = BatchNorm()
        D_1_7 = bn(D_1_7)
        D_1_7 = tf.nn.relu(D_1_7)

    """
        #如果是最后一层的话就这样
        D_7_shape = D_1_7.get_shape().as_list()[-1]

        with tf.variable_scope('D_1_8'):
            filter = create_variables(name='DRN_1_8_var', shape=[1, 1, D_7_shape, 1])
            D_1_8 = tf.nn.atrous_conv2d(value=D_1_7, filters=filter, rate=1, padding='SAME')
            D_8_1=D_1_8
            D_8 = tf.nn.relu(D_1_8)

    """

    # second block
    with tf.variable_scope('D_2_1'):
        filter = create_variables(name='DRN_2_1_var', shape=[9, 9, input_channel, 16])
        # D_2_1 = tf.nn.atrous_conv2d(value=input_layer, filters=filter, rate=1, padding='SAME')
        D_2_1 = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_2_1 = tf.nn.bias_add(D_2_1, biases)
        bn = BatchNorm()
        D_2_1 = bn(D_2_1)
        D_2_1 = tf.nn.relu(D_2_1)

    D_1_shape = D_2_1.get_shape().as_list()[-1]

    with tf.variable_scope('D_2_2'):
        filter = create_variables(name='DRN_2_2_var', shape=[7, 7, D_1_shape, 24])
        # D_2_2 = tf.nn.atrous_conv2d(value=D_2_1, filters=filter, rate=1, padding='SAME')
        D_2_2 = tf.nn.conv2d(D_2_1, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0))

        D_2_2 = tf.nn.bias_add(D_2_2, biases)
        bn = BatchNorm()
        D_2_2 = bn(D_2_2)
        D_2_2 = tf.nn.relu(D_2_2)

    D_2_shape = D_2_2.get_shape().as_list()[-1]

    """
            池化层第一层
    """
    with tf.variable_scope('D_2_2_pool'):
        D_2_2 = tf.nn.max_pool(D_2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
                               )

    with tf.variable_scope('D_2_3'):
        filter = create_variables(name='DRN_2_3_var', shape=[5, 5, D_2_shape, 32])
        # D_2_3 = tf.nn.atrous_conv2d(value=D_2_2, filters=filter, rate=1, padding='SAME')
        D_2_3 = tf.nn.conv2d(D_2_2, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0))

        D_2_3 = tf.nn.bias_add(D_2_3, biases)
        bn = BatchNorm()
        D_2_3 = bn(D_2_3)
        D_2_3 = tf.nn.relu(D_2_3)

    D_3_shape = D_2_3.get_shape().as_list()[-1]

    with tf.variable_scope('D_2_4'):
        filter = create_variables(name='DRN_2_4_var', shape=[5, 5, D_3_shape, 32])
        # D_2_4 = tf.nn.atrous_conv2d(value=D_2_3, filters=filter, rate=1, padding='SAME')
        D_2_4 = tf.nn.conv2d(D_2_3, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0))

        D_2_4 = tf.nn.bias_add(D_2_4, biases)
        bn = BatchNorm()
        D_2_4 = bn(D_2_4)
        D_2_4 = tf.nn.relu(D_2_4)

    D_4_shape = D_2_4.get_shape().as_list()[-1]

    """
        池化层第二层
    """
    with tf.variable_scope('D_2_4_pool'):
        D_2_4 = tf.nn.max_pool(D_2_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
                               )

    with tf.variable_scope('D_2_5'):
        filter = create_variables(name='DRN_2_5_var', shape=[3, 3, D_4_shape, 32], stddev=0.001)
        # D_2_5 = tf.nn.atrous_conv2d(value=D_2_4, filters=filter, rate=1, padding='SAME')
        D_2_5 = tf.nn.conv2d(D_2_4, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0))

        D_2_5 = tf.nn.bias_add(D_2_5, biases)
        bn = BatchNorm()
        D_2_5 = bn(D_2_5)
        D_2_5 = tf.nn.relu(D_2_5)

    D_5_shape = D_2_5.get_shape().as_list()[-1]

    with tf.variable_scope('D_2_6'):
        filter = create_variables(name='DRN_2_6_var', shape=[3, 3, D_5_shape, 16], stddev=0.001)
        # D_2_6 = tf.nn.atrous_conv2d(value=D_2_5, filters=filter, rate=1, padding='SAME')
        D_2_6 = tf.nn.conv2d(D_2_5, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_2_6 = tf.nn.bias_add(D_2_6, biases)
        bn = BatchNorm()
        D_2_6 = bn(D_2_6)
        D_2_6 = tf.nn.relu(D_2_6)

    D_6_shape = D_2_6.get_shape().as_list()[-1]

    with tf.variable_scope('D_2_7'):
        filter = create_variables(name='DRN_2_7_var', shape=[3, 3, D_6_shape, 16], stddev=0.001)
        # D_2_7 = tf.nn.atrous_conv2d(value=D_2_6, filters=filter, rate=1, padding='SAME')
        D_2_7 = tf.nn.conv2d(D_2_6, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_2_7 = tf.nn.bias_add(D_2_7, biases)
        bn = BatchNorm()
        D_2_7 = bn(D_2_7)
        D_2_7 = tf.nn.relu(D_2_7)

    # third block
    with tf.variable_scope('D_3_1'):
        filter = create_variables(name='DRN_3_1_var', shape=[7, 7, input_channel, 16])
        # D_3_1 = tf.nn.atrous_conv2d(value=input_layer, filters=filter, rate=1, padding='SAME')
        D_3_1 = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_3_1 = tf.nn.bias_add(D_3_1, biases)
        bn = BatchNorm()
        D_3_1 = bn(D_3_1)
        D_3_1 = tf.nn.relu(D_3_1)

    D_1_shape = D_3_1.get_shape().as_list()[-1]

    with tf.variable_scope('D_3_2'):
        filter = create_variables(name='DRN_3_2_var', shape=[5, 5, D_1_shape, 24])
        # D_3_2 = tf.nn.atrous_conv2d(value=D_3_1, filters=filter, rate=1, padding='SAME')
        D_3_2 = tf.nn.conv2d(D_3_1, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0))

        D_3_2 = tf.nn.bias_add(D_3_2, biases)
        bn = BatchNorm()
        D_3_2 = bn(D_3_2)
        D_3_2 = tf.nn.relu(D_3_2)

    D_2_shape = D_3_2.get_shape().as_list()[-1]

    """
            池化层第一层
    """
    with tf.variable_scope('D_3_2_pool'):
        D_3_2 = tf.nn.max_pool(D_3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
                               )

    with tf.variable_scope('D_3_3'):
        filter = create_variables(name='DRN_3_3_var', shape=[3, 3, D_2_shape, 48])
        # D_3_3 = tf.nn.atrous_conv2d(value=D_3_2, filters=filter, rate=1, padding='SAME')
        D_3_3 = tf.nn.conv2d(D_3_2, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [48], tf.constant_initializer(0))

        D_3_3 = tf.nn.bias_add(D_3_3, biases)
        bn = BatchNorm()
        D_3_3 = bn(D_3_3)
        D_3_3 = tf.nn.relu(D_3_3)

    D_3_shape = D_3_3.get_shape().as_list()[-1]

    with tf.variable_scope('D_3_4'):
        filter = create_variables(name='DRN_3_4_var', shape=[3, 3, D_3_shape, 48])
        # D_3_4 = tf.nn.atrous_conv2d(value=D_3_3, filters=filter, rate=1, padding='SAME')
        D_3_4 = tf.nn.conv2d(D_3_3, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [48], tf.constant_initializer(0))

        D_3_4 = tf.nn.bias_add(D_3_4, biases)
        bn = BatchNorm()
        D_3_4 = bn(D_3_4)
        D_3_4 = tf.nn.relu(D_3_4)

    D_4_shape = D_3_4.get_shape().as_list()[-1]
    """
            池化层第二层
    """
    with tf.variable_scope('D_3_4_pool'):
        D_3_4 = tf.nn.max_pool(D_3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
                               )

    with tf.variable_scope('D_3_5'):
        filter = create_variables(name='DRN_3_5_var', shape=[3, 3, D_4_shape, 48], stddev=0.001)
        # D_3_5 = tf.nn.atrous_conv2d(value=D_3_4, filters=filter, rate=1, padding='SAME')
        D_3_5 = tf.nn.conv2d(D_3_4, filter, strides=[1, 1, 1, 1], padding='SAME')

        biases = _variable_on_cpu('biases', [48], tf.constant_initializer(0))

        D_3_5 = tf.nn.bias_add(D_3_5, biases)
        bn = BatchNorm()
        D_3_5 = bn(D_3_5)
        D_3_5 = tf.nn.relu(D_3_5)

    D_5_shape = D_3_5.get_shape().as_list()[-1]

    with tf.variable_scope('D_3_6'):
        filter = create_variables(name='DRN_1_6_var', shape=[3, 3, D_5_shape, 24], stddev=0.001)
        # D_3_6 = tf.nn.atrous_conv2d(value=D_3_5, filters=filter, rate=1, padding='SAME')
        D_3_6 = tf.nn.conv2d(D_3_5, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0))

        D_3_6 = tf.nn.bias_add(D_3_6, biases)
        bn = BatchNorm()
        D_3_6 = bn(D_3_6)
        D_3_6 = tf.nn.relu(D_3_6)

    D_6_shape = D_3_6.get_shape().as_list()[-1]

    with tf.variable_scope('D_3_7'):
        filter = create_variables(name='DRN_3_7_var', shape=[3, 3, D_6_shape, 24], stddev=0.001)
        # D_3_7 = tf.nn.atrous_conv2d(value=D_3_6, filters=filter, rate=1, padding='SAME')
        D_3_7 = tf.nn.conv2d(D_3_6, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0))

        D_3_7 = tf.nn.bias_add(D_3_7, biases)
        bn = BatchNorm()
        D_3_7 = bn(D_3_7)
        D_3_7 = tf.nn.relu(D_3_7)

    # 进行链接
    D_7 = tf.concat([D_1_7, D_2_7, D_3_7], 3, name='concat_2')
    print('D_7 shape:', D_7)

    # 如果是最后一层的话就这样
    D_7_shape = D_7.get_shape().as_list()[-1]

    with tf.variable_scope('D_8'):
        filter = create_variables(name='DRN_8_var', shape=[1, 1, D_7_shape, 16], stddev=0.001)
        # D_8 = tf.nn.atrous_conv2d(value=D_7, filters=filter, rate=1, padding='SAME')
        D_8 = tf.nn.conv2d(D_7, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))

        D_8 = tf.nn.bias_add(D_8, biases)
        bn = BatchNorm()
        D_8 = bn(D_8)
        D_8_2 = D_8
        # 同时利用relu和sigmoid的激活函数
        D_8 = tf.nn.relu(tf.nn.sigmoid(D_8))
        D_8 = tf.depth_to_space(D_8, 4)
        print("最终结果的维度：", D_8.get_shape())

    return D_8, D_8_2, D_2_7

    # second block

    # third block


def inference_test(input_layer):
    input_channel = input_layer.get_shape().as_list()[-1]
    input_w = input_layer.get_shape().as_list()[2]
    input_h = input_layer.get_shape().as_list()[1]
    with tf.variable_scope('D_1'):
        filter = create_variables(name='D_1_va', shape=[3, 3, input_channel, 16])
        DRN_1 = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
        DRN_1 = tf.nn.relu(DRN_1)

    with tf.variable_scope('D_2'):
        filter = create_variables(name='D_2_va', shape=[3, 3, 16, 32])
        DRN_2 = tf.nn.conv2d(DRN_1, filter, strides=[1, 1, 1, 1], padding='SAME')
        DRN_2 = tf.nn.relu(DRN_2)

    with tf.variable_scope('D_3'):
        filter = create_variables(name='D_3_va', shape=[3, 3, 32, 64])
        DRN_3 = tf.nn.conv2d(DRN_2, filter, strides=[1, 1, 1, 1], padding='SAME')
        DRN_3 = tf.nn.relu(DRN_3)

    with tf.variable_scope('D_4'):
        filter = create_variables(name='D_4_va', shape=[3, 3, 64, 1])
        DRN_4 = tf.nn.conv2d(DRN_3, filter, strides=[1, 1, 1, 1], padding='SAME')
        DRN_41 = DRN_4
        DRN_4 = tf.nn.sigmoid(DRN_4)

    return DRN_4, DRN_41, DRN_3


# 进行整体网络架构的搭建
# 输入的input的维度为[batch_size,h,w,c]
def inference(input_layer):
    # 下面按照那个结构进行相应的搭建框架的结构如下所示：[batch,h,w,3]出去的时候也应该是[batch,h,w,3]

    """
    这下面的数字以后会改的
    输入图片进行相应的训练


    """
    input_channel = input_layer.get_shape().as_list()[-1]
    input_w = input_layer.get_shape().as_list()[2]
    input_h = input_layer.get_shape().as_list()[1]






    # [7,7,16]
    with tf.variable_scope('DRN_1'):
        filter = create_variables(name='DRN_1_va', shape=[7, 7, input_channel, 16])
        DRN_1 = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0))
        DRN_1 = tf.nn.bias_add(DRN_1, biases)

        bn = BatchNorm()
        DRN_1 = bn(DRN_1)
        DRN_1 = tf.nn.relu(DRN_1)

    with tf.variable_scope('DRN_2'):
        DRN_2 = residual_block(DRN_1, 16, dilation=1, stride=1, first_block=False)

    """

        进行第一次反卷积

        """
    DRN_2_output_shape = DRN_2.get_shape().as_list()[-1]

    # 反卷积过程反卷积
    with tf.variable_scope('DRN_2_output'):
        filter = create_variables(name='DRN_1_output_va', shape=[7, 7, 1, DRN_2_output_shape])
        DRN_2_output = tf.nn.conv2d_transpose(DRN_2, filter, [batch_size, 240, 240, 1], strides=[1, 1, 1, 1],
                                              padding='SAME')
        DRN_2_output = tf.nn.relu(DRN_2_output)

    # 调用了残差网络块进行相应的设计
    # [3,3,16]+[3,3,16]的残差网络的设计

    # 池化层
    with tf.variable_scope('DRN_2_pooling'):
        DRN_2 = tf.nn.max_pool(DRN_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # [3,3,32]+[3,3,32]的残差网络的设,stride设计为2相当于进行了降采样的操作
    with tf.variable_scope('DRN_3'):
        DRN_3 = residual_block(DRN_2, 32, dilation=1, stride=1, first_block=False)

    """

    进行第二次反卷积

    """
    """
    DRN_3_output_shape = DRN_3.get_shape().as_list()[-1]
    with tf.variable_scope('DRN_3_output'):
        filter = create_variables(name='DRN_3_output_va', shape=[3, 3, DRN_3_output_shape, DRN_3_output_shape])
        DRN_3_output=tf.nn.conv2d_transpose(DRN_3,filter,[batch_size,240,240,DRN_3_output_shape],strides=[1,2,2,1],padding='SAME')


        DRN_3=tf.nn.relu(DRN_3_output)

    """

    # 池化层
    with tf.variable_scope('DRN_3_pooling'):
        DRN_3 = tf.nn.max_pool(DRN_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # [3,3,64]+[3,3,64]的残差网络的设,stride设计为2相当于进行了降采样的操作，
    # 这里还涉及到了risdual的设计细节关于输入尺寸和输出尺寸不一致的问题
    with tf.variable_scope('DRN_4_1'):
        DRN_4 = residual_block(DRN_3, 64, dilation=1, stride=1, first_block=False)

    """

        进行第三次反卷积

    """
    """
    DRN_4_output_shape = DRN_4.get_shape().as_list()[-1]
    with tf.variable_scope('DRN_4_output'):
        filter = create_variables(name='DRN_4_output_va', shape=[3, 3, DRN_4_output_shape, DRN_4_output_shape])
        DRN_4_output = tf.nn.conv2d_transpose(DRN_4, filter, [batch_size, 240, 240, DRN_4_output_shape], strides=[1, 2, 2, 1],
                                              padding='SAME')
        DRN_4_output = tf.nn.relu(DRN_4_output)
    """

    # [3,3,64]+[3,3,64]的残差网络的设,stride设计为2相当于进行了降采样的操作
    with tf.variable_scope('DRN_4_2'):
        DRN_4 = residual_block(DRN_4, 64, dilation=1, stride=1, first_block=False)

    # 池化层
    with tf.variable_scope('DRN_4_pooling'):
        DRN_4 = tf.nn.max_pool(DRN_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    """
    反卷积层

    """
    """
    DRN_4_output_shape = DRN_4.get_shape().as_list()[-1]
    with tf.variable_scope('DRN_41_output'):
        filter = create_variables(name='DRN_41_output_va', shape=[3, 3, 1, DRN_4_output_shape])
        DRN_4_output = tf.nn.conv2d_transpose(DRN_4, filter, [batch_size, 240, 240, 1],
                                              strides=[1, 2, 2, 1],
                                              padding='SAME')
        DRN_4 = tf.nn.relu(DRN_4_output)

    """

    """

    进入这里的时候就是原图大小了相当于没有进行池化层的操作

    """
    # [3,3,128]+[3,3,128]的残差网络的设,stride设计为2相当于进行了降采样的操作
    with tf.variable_scope('DRN_5_1'):
        DRN_5 = residual_block(DRN_4, 128, dilation=1, stride=1, first_block=False)

    with tf.variable_scope('DRN_5_2'):
        DRN_5 = residual_block(DRN_5,128, dilation=1, stride=1, first_block=False)

    # [3,3,256]+[3,3,256]的残差网络的设,stride设计为2相当于进行了降采样的操作
    with tf.variable_scope('DRN_6_1'):
        DRN_6 = residual_block(DRN_5, 128, dilation=2, stride=1, first_block=False)

    with tf.variable_scope('DRN_6_2'):
        DRN_6 = residual_block(DRN_5, 256, dilation=2, stride=1, first_block=False)

    # [3,3,256]+[3,3,256]的残差网络的设,stride设计为2相当于进行了降采样的操作
    with tf.variable_scope('DRN_7_1'):
        DRN_7 = residual_block_1(DRN_6, 512, dilation=5, stride=1, first_block=False)

    with tf.variable_scope('DRN_7_2'):
        DRN_7 = residual_block_1(DRN_7, 512, dilation=5, stride=1, first_block=False)

    DRN_7_shape = DRN_7.get_shape().as_list()[-1]
    # [3,3,512]的正常网络模型,dilated设计为1相当于进行了降采样的操作
    with tf.variable_scope('DRN_8_1'):
        filter = create_variables(name='DRN_8_1_var', shape=[3, 3, DRN_7_shape, 256])
        DRN_8 = tf.nn.atrous_conv2d(value=DRN_7, filters=filter, rate=2, padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0))
        DRN_8 = tf.nn.bias_add(DRN_8, biases)

        bn = BatchNorm()
        DRN_8 = bn(DRN_8)
        DRN_8 = tf.nn.relu(DRN_8)

    DRN_8_shape = DRN_8.get_shape().as_list()[-1]

    with tf.variable_scope('DRN_8_2'):
        filter = create_variables(name='DRN_8_2_var', shape=[3, 3, DRN_8_shape, 256])
        DRN_8 = tf.nn.atrous_conv2d(value=DRN_8, filters=filter, rate=2, padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0))
        DRN_8 = tf.nn.bias_add(DRN_8, biases)

        bn = BatchNorm()
        DRN_8 = bn(DRN_8)
        DRN_8 = tf.nn.relu(DRN_8)

    DRN_8_shape = DRN_8.get_shape().as_list()[-1]
    # [3,3,512]的正常网络模型,dilated设计为1相当于进行了降采样的操作
    with tf.variable_scope('DRN_9_1'):
        filter = create_variables(name='DRN_9_1_var', shape=[3, 3, DRN_8_shape, 256])
        DRN_9 = tf.nn.atrous_conv2d(value=DRN_8, filters=filter, rate=1, padding='SAME')


        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0))
        DRN_9 = tf.nn.bias_add(DRN_9, biases)

        bn = BatchNorm()
        DRN_9 = bn(DRN_9)
        DRN_9 = tf.nn.relu(DRN_9)

    DRN_9_shape = DRN_9.get_shape().as_list()[-1]

    with tf.variable_scope('DRN_9_2'):
        filter = create_variables(name='DRN_9_2_var', shape=[3, 3, DRN_9_shape, 128])
        DRN_9 = tf.nn.atrous_conv2d(value=DRN_9, filters=filter, rate=1, padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0))
        DRN_9 = tf.nn.bias_add(DRN_9, biases)

        bn = BatchNorm()
        DRN_9 = bn(DRN_9)
        DRN_9 = tf.nn.relu(DRN_9)

    # 最终得到的模型大小为[n/8,n/8,512],所以为了还原成原始图像的大小
    DRN_9_shape = DRN_9.get_shape().as_list()[-1]
    with tf.variable_scope('DRN_10'):
        filter = create_variables(name='DRN_10_var', shape=[1, 1, DRN_9_shape, 64])
        DRN_10 = tf.nn.conv2d(DRN_9, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0))
        DRN_10 = tf.nn.bias_add(DRN_10, biases)

        bn = BatchNorm()
        DRN_10 = bn(DRN_10)

        DRN_10 = DRN_10
        DRN_101 = DRN_10
        DRN_10 = tf.nn.relu(tf.nn.sigmoid(DRN_10))

    # 使用reshape的操作变换形状的操作，现在的形状为[1,n/8,n/8,64]==>[1,n,n,1]，这里面的reshape 的操作可能需要改一下,首先现在是没有一批的操作
    # 现在假设还没有批处理的进行
    values = []
    # 以前的input_w应该被处理为能被8整除的数字了，利用填充技术把剩余的部分填充成0
    print("开始进行变换维度的操作")
    counts = 0
    print(DRN_10, input_h // 8, input_w // 8)
    """
    hh=input_h//8
    ww=input_w//8

    for i in tf.split(DRN_10,hh,1):
        counts=counts+1
        for ii in tf.split(i,ww,2):

            iii=tf.reshape(ii,[batch_size,8,8,1])
            values.append(iii)
            ccc=tf.concat(values,2)
    print("进行维度变化",counts)

    result_1=tf.split(ccc ,ww,2)
    result=tf.concat(result_1,1)
    #print("最终结果的维度：",result.get_shape())
    """
    # result=DRN_10
    # 不使用这一层，因为效果不好
    print("最终结果的维度：", DRN_10.get_shape())
    result = tf.depth_to_space(DRN_10, 8)
    print("最终结果的维度：", result.get_shape())
    # 返回最终的结果
    return result, DRN_101, DRN_1
    # return result,DRN_101,DRN_4


"""   
if __name__=="__main__":
    #进行相应的网络设计过程
    main()
"""








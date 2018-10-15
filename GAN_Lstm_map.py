"""
作者：赵厚涛
时间：2018.9.4
备注：非卖品


"""

#系统模块
import re
#深度学习的模型库
import tensorflow as tf
#载入项目模块
from PIL import Image
import random
import pickle as p
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
import numpy as np
from dilated_resnet import batch_size
from cnn_basenet import *

a=tf.constant(1,shape=[1,2])

"""
利用强化学习和GAN学习对于得到的深度图进行处理从而使得它能够在原来的基础上可以
得到比较好的结果
"""
"""
    1.先在小数据集上进性预训练的过程
    2.然后转到大数据集上进行训练过程
    3.通过GAN和LSTM上进行结合，改进基础网络得到的特征图
    4.想想怎么把强化学习加进去
    """


#先是进行残差网络的建立
def multi_residual_block(input_tensor,name):
    with tf.variable_scope(name):
        for i in range(3):
            output = residual_block(input_tensor, output_channel=2**(i+6),dilation=1, stride=1,
                                    first_block=False,name='residual_{:d}'.format(i + 1))
            #print("每一次得到的结果的维度：",output.get_shape() )
            input_tensor = output

    """
    得到的结果为[batch_size,h/8,w/8,256],下一步要将他转换成[batch_size,h,w,32]
    """
    #这里将会用到xception的方法？
    """
    进行一些结果的简化过程
    """



    #print("开始结果的维度：", output.get_shape())
    output = tf.depth_to_space(output, 8)
    #print("中间结果的维度：", output.get_shape())
    change_shape = output.get_shape().as_list()[-1]
    with tf.variable_scope("second_%s"%name):
        filter=create_variables(name='change_d',shape=[1,1,change_shape,32])
        output=tf.nn.conv2d(output,filter,strides=[1,1,1,1],name='change_shape_%s'%name, padding='SAME')
        biases = variable_on_cpu('biases_change_shape_%s'%name, [32], tf.constant_initializer(0))
        conv1 = tf.nn.bias_add(output, biases)

        bn = BatchNorm()
        conv1 = bn(conv1)

    conv1 = tf.nn.relu(conv1)
    print("最终结果的维度：", output.get_shape())
    return output

def build_attentive_rnn(input_tensor):
    """
    Generator的attentive recurrent部分, 主要是为了找到attention部分
    :param input_tensor:
    :param name:
    :return:
    """
    [batch_size, tensor_h, tensor_w, _] = input_tensor.get_shape().as_list()
    with tf.variable_scope('attentive_inference'):
        init_attention_map = tf.constant(0.5, dtype=tf.float32,
                                         shape=[batch_size, tensor_h, tensor_w, 1])
        init_cell_state = tf.constant(0.0, dtype=tf.float32,
                                      shape=[batch_size, tensor_h, tensor_w, 32])
        init_lstm_feats = tf.constant(0.0, dtype=tf.float32,
                                      shape=[batch_size, tensor_h, tensor_w, 32])

        attention_map_list = []

        for i in range(4):
            #输入一个240x240的图片，作为input_tensor,并且可以
            attention_input = tf.concat((input_tensor, init_attention_map), axis=-1)
            conv_feats = multi_residual_block(input_tensor=attention_input,
                                              name='multi_residual_block_{:d}'.format(i + 1))
            """
            这里输出的是步长为
            
            """
            lstm_ret =conv_lstm(input_tensor=conv_feats,
                                       input_cell_state=init_cell_state,
                                       name='conv_lstm_block_{:d}'.format(i + 1))
            # 输出的是[batch_size,h,w,1]
            init_attention_map = lstm_ret['attention_map']
            init_cell_state = lstm_ret['cell_state']
            init_lstm_feats = lstm_ret['lstm_feats']

            attention_map_list.append(lstm_ret['attention_map'])

    ret = {
        'final_attention_map': init_attention_map,
        'final_lstm_feats': init_lstm_feats,
        'attention_map_list': attention_map_list
    }

    return ret

def compute_attentive_rnn_loss(input_tensor, label_tensor, name):
    """
    计算attentive rnn损失
    :param input_tensor:
    :param label_tensor:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        #输出的大小为[batch_size,h,w,1]
        inference_ret =build_attentive_rnn(input_tensor=input_tensor
                                                 )
        loss = tf.constant(0.0, tf.float32)
        n = len(inference_ret['attention_map_list'])
        for index, attention_map in enumerate(inference_ret['attention_map_list']):
            #分出了重要程度
            """
            mse_loss = tf.pow(0.8, n - index + 1) * \
                       tf.losses.mean_squared_error(labels=label_tensor,
                                                    predictions=attention_map)
            """
            mse_loss = tf.pow(0.8, n - index + 1) * \
                       tf.reduce_mean(tf.square(label_tensor-attention_map))
            loss = tf.add(loss, mse_loss)
        loss=loss/batch_size

    return loss, inference_ret['final_attention_map']


"""
def GAN_reinforcement(predict_map):
    print("START WORKING！")
    #怎么得到的

"""



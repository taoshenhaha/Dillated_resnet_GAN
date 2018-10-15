import tensorflow as tf




class BatchNorm(object):
    """
    BN操作类
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
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


def variable_on_cpu(name, shape, initializer):
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
    var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

#在这里进行相应的修改操作吧？
def conv_bn_relu_layer(input_layer,filter_shape,stride,dilation=1,name=None):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)
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
    biases = variable_on_cpu('biases_%s'%name, [out_channel], tf.constant_initializer(0))
    conv_layer = tf.nn.bias_add(conv_layer, biases)
    with tf.variable_scope('batch_normal_%s'%name):
        bn = BatchNorm(name=name)
        bn_layer = bn(conv_layer)

    #bn_layer = batch_normalization_layer(, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


# 开始构造block的结构过程,增加几层残差网络,为5层
def residual_block(input_layer,output_channel, dilation=1, stride=1, first_block=False,name="residual_b" ):
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
    with tf.variable_scope('%s_conv1_in_block'%name):
        with tf.variable_scope('Batch_norma_%s'%name):

            if first_block:
                filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
                conv1 = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
                biases = variable_on_cpu('biases', [output_channel], tf.constant_initializer(0))
                conv1 = tf.nn.bias_add(conv1, biases)

                bn = BatchNorm()
                conv1 = bn(conv1)

                conv1 = tf.nn.relu(conv1)
            else:
                conv1 = conv_bn_relu_layer(input_layer, [3, 3, input_channel, output_channel],
                                           stride, dilation,name='%s_conv1_in_block'%name)

    with tf.variable_scope('%s_conv2_in_block'%name):
        """
        这里进行相应的步数为2的情况
        
        """
        conv2 = conv_bn_relu_layer(conv1, [3, 3, output_channel, output_channel],2, dilation,name='%s_conv2_in_block'%name)

    # mid_layer=bn_relu_conv_layer(input_layer,[3,3],stride,dilation=dilation)

    # out_layer=bn_relu_conv_layer(mid_layer,[3,3],stride,dilation=dilation)


    with tf.variable_scope('Unichannel_%s'%name):
        if input_channel != output_channel:
            filter = create_variables(name='Unichannel', shape=[1, 1, input_channel, output_channel])
            """
            这里的stride也为2
            """
            input_layer = tf.nn.conv2d(input_layer, filter, strides=[1, 2, 2, 1], padding='SAME')
            biases = variable_on_cpu('biases', [output_channel], tf.constant_initializer(0))
            input_layer = tf.nn.bias_add(input_layer, biases)

    result = tf.add(input_layer, conv2)
    with tf.variable_scope('Batch_norma_%s'%name):
        bn = BatchNorm()
        result = bn(result)

    result = tf.nn.relu(result)
    """
    输出的结果为[batch_size,h/2,w/2,channel]
    """
    return result


#需要一些比较好的LSTM的算法技巧
def conv_lstm(input_tensor, input_cell_state, name):
    """
    attentive recurrent net中的convolution lstm 见公式(3)
    :param input_tensor:
    :param input_cell_state:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        #输入的是[batch_size,h,w,channel]
        input_channel=input_tensor.get_shape().as_list()[-1]

        """
        channel=32可以转换成16
        """
        filter=create_variables(name='aconv_i',shape=[3,3,input_channel,32])
        conv_i=tf.nn.conv2d(input_tensor, filter, strides=[1, 1, 1, 1], padding='SAME', name='conv_i')
        #这里加入了一些偏差结果
        biases = variable_on_cpu('biases_i', [32], tf.constant_initializer(0))
        conv_i = tf.nn.bias_add(conv_i, biases)


        """
        conv_i = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                             stride=1, use_bias=False, name='conv_i')
        """
        sigmoid_i = tf.nn.sigmoid(conv_i, name='sigmoid_i')
        filter = create_variables(name='aconv_f', shape=[3, 3, input_channel, 32])
        conv_f = tf.nn.conv2d(input_tensor, filter, strides=[1, 1, 1, 1], padding='SAME', name='conv_f')
        biases = variable_on_cpu('biases_f', [32], tf.constant_initializer(0))
        conv_f = tf.nn.bias_add(conv_f, biases)
        """
        conv_f = self.conv2d(inputdata=input_tensor, out_channel=32, kernel_size=3, padding='SAME',
                             stride=1, use_bias=False, name='conv_f')
        """
        #输出的是[batch_size,h,w,channel]
        sigmoid_f = tf.nn.sigmoid(conv_f, name='sigmoid_f')

        filter = create_variables(name='aconv_c', shape=[3, 3, input_channel, 32])
        conv_c = tf.nn.conv2d(input_tensor, filter, strides=[1, 1, 1, 1], padding='SAME',name='conv_c')
        biases = variable_on_cpu('biases_c', [32], tf.constant_initializer(0))
        conv_c = tf.nn.bias_add(conv_c, biases)


        cell_state = sigmoid_f * input_cell_state + \
                     sigmoid_i * tf.nn.tanh(conv_c)
        filter = create_variables(name='aconv_o', shape=[3, 3, input_channel, 32])
        conv_o = tf.nn.conv2d(input_tensor, filter, strides=[1, 1, 1, 1], padding='SAME',name='conv_o')
        biases = variable_on_cpu('biases_o', [32], tf.constant_initializer(0))
        conv_o = tf.nn.bias_add(conv_o, biases)

        sigmoid_o = tf.nn.sigmoid(conv_o, name='sigmoid_o')
        # 输出的是[batch_size,h,w,channel]
        lstm_feats = sigmoid_o * tf.nn.tanh(cell_state)
        filter = create_variables(name='conv_attention', shape=[3, 3, input_channel, 1])
        attention_map = tf.nn.conv2d(lstm_feats, filter, strides=[1, 1, 1, 1], padding='SAME',name='attention_map')
        biases = variable_on_cpu('biases_attention', [1], tf.constant_initializer(0))
        attention_map = tf.nn.bias_add(attention_map, biases)

        """
        attention_map = tf.nn.conv2d(inputdata=lstm_feats, out_channel=1, kernel_size=3, padding='SAME',
                                    stride=1, use_bias=False, name='attention_map')
        """
        # 输出的是[batch_size,h,w,1]
        attention_map = tf.nn.sigmoid(attention_map)

        ret = {
            'attention_map': attention_map,
            'cell_state': cell_state,
            'lstm_feats': lstm_feats
        }

        return ret









"""
时间：2018.9.5
作者：赵厚涛




"""
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import scipy
import glob
import os
from dilated_resnet import *
from GAN_Lstm_map import *
tf.reset_default_graph()
Zoom_size=1
"""
Image_File_dir = "output_test/crop_image"
Label_File_dir = "label"
"""


ckpt_dir = "./model"

"""
Image_File_dir = "output/crop_image"
Label_File_dir = "output/crop_groundtruth"
"""


"""
Image_File_dir = "D:/Deeplearning_Demo/TensorFlow_demo/Ours_crowd_counting/output/crop_image"
Label_File_dir = "D:/Deeplearning_Demo/TensorFlow_demo/Ours_crowd_counting/output/crop_groundtruth"
"""
Image_list = []
Label_list = []


each_step = 10000

"""
接下来就是进行输入的操作了

"""
#lena = np.array(ndimage.imread("test_image/IMGRGB_5.jpg",flatten=False) )# 读取和代码处于同一目录下的 lena.png
lena = np.array(ndimage.imread("output/crop_image/IMG_24_1.jpg",flatten=False) )# 读取和代码处于同一目录下的 lena.png

# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
print(lena.shape) #(512, 512, 3)
print(lena.dtype)

plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
lena=tf.expand_dims(lena,0)

#载入标签


#label_a=np.load("test_label/IMG_output_169_5.npy" )
label_a=np.load("output/crop_groundtruth/IMG_output_24_1.npy" )
print("标签为：",label_a.dtype )
counta=np.sum(label_a)
print("真实人数是：",counta)
i1 = Image.fromarray(label_a*10000)
# plt.figure(i+2)
result1 = i1.show()






"""
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
#图像数据
"""
image = tf.read_file(file_queue[0])  # reader读取序列
image = tf.image.decode_jpeg(image, channels=3)  # 解码，tensor
image = tf.image.resize_images(image, [240, 240])
#image = tf.image.per_image_standardization(image)
"""
#标签数据

"""

label = tf.read_file(file_queue[1])  # reader读取序列

# 读出的 value 是 string，现在转换为 uint8 型的向量
record_bytes = tf.decode_raw(label, tf.float32)

depth_major = tf.reshape(tf.slice(record_bytes, [32], [240 * 240 * 1]),
                         [1, 240, 240])  # depth, height, width
print("done")
uint8image = tf.transpose(depth_major, [1, 2, 0])

label = tf.cast(uint8image, tf.float32) / (Zoom_size)

#这里需要用参数进行设计
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=2,
                                          num_threads=1,
                                          capacity=10000)


"""

image_batch=tf.placeholder(tf.float32,shape=[1,240,240,3])
label_batch=tf.placeholder(tf.float32,shape=[1,240,240,1])
#改变函数
#predict_map1, D_1, D_2=inference(image_batch)
#predict_map_test=build_attentive_rnn(image_batch)
total_loss,predict_map_test=compute_attentive_rnn_loss(image_batch, label_batch, name="total_loss")
predict_map=predict_map_test
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 占用GPU40%的显存
with tf.Session(config=config) as sess:

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())


    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")



    print("Limain ", lena)
    print("Limain ",lena.dtype)
    lena=lena.eval()




    predict_map=sess.run(predict_map,feed_dict={image_batch:lena})
    result_groundtruth = tf.squeeze(predict_map[0, :, :, :]).eval()

    # print("得到的结果是：", result_groundtruth.dtype)
    # 为什么得到的数据会那么的大呢？
    countb = np.sum(result_groundtruth)
    print("预测人数是：", countb)
    print_result_groundtruth = result_groundtruth * 10000

    i1 = Image.fromarray(print_result_groundtruth)
    # plt.figure(i+2)
    result1 = i1.show()
    #a=sess.run(label_batch)












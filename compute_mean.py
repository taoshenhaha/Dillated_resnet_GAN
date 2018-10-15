
import tensorflow as tf
import PIL.Image as Image
import numpy as np
from scipy import ndimage
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
#Train_image=267
Train_image=10
tf.reset_default_graph()
mean=0
sample_num=50
#file_dir='ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images/IMG_%s.jpg'%countss'

R=0
G=0
B=0
count=0

for count in range(Train_image):
    count = count + 1
    print("进度条为：", count, "/", Train_image)


    # Image_list.append(file_dir + '/' + file)
    # i=i+1
    # print("i:",i)
    # i = i + 1
    #count=count+1
    RGB = np.array(ndimage.imread('ShanghaiTech_Crowd_Counting_Dataset/part_A_final/'
                                        'train_data/images/IMG_%s.jpg' % count) )


    RGB_W=RGB.shape[1]
    RGB_H=RGB.shape[0]
    print(RGB_W,RGB_H)
    # print(RGB)
    # RGB=RGB.astype(np.float32)
    R_a = np.sum(RGB[:, :, 0]) / (RGB_W * RGB_H)
    G_a = np.sum(RGB[:, :, 1]) / (RGB_W * RGB_H)
    B_a = np.sum(RGB[:, :, 2]) / (RGB_W * RGB_H)
    R=R_a+R
    G=G_a+G
    B=B_a+B
mean_R=R/count
mean_G=G/count
mean_B=B/count
print("R:",mean_R)
print("G:",mean_G)
print("B:",mean_B)

for count in range(Train_image):
    count = count + 1
    print("进度条为：", count, "/", Train_image)

    # Image_list.append(file_dir + '/' + file)
    # i=i+1
    # print("i:",i)
    # i = i + 1

    """
    RGB = np.array(ndimage.imread('ShanghaiTech_Crowd_Counting_Dataset/part_A_final/'
                                        'train_data/images/IMG_%s.jpg' % count))
    """

    RGB = np.array(ndimage.imread('test_image/IMG_169_%s.jpg' % count))
    # print(RGB)
    # RGB=RGB.astype(np.float32)
    #106.  95.  92.
    """
    RGB[:, :, 0]= RGB[:, :, 0]-mean_R
    RGB[:, :, 1]= RGB[:, :, 1]-mean_G
    RGB[:, :, 2] = RGB[:, :, 2]-mean_B
    """
    RGB[:, :, 0] = RGB[:, :, 0] - 106.
    RGB[:, :, 1] = RGB[:, :, 1] - 95.
    RGB[:, :, 2] = RGB[:, :, 2] - 92.
    RGB = Image.fromarray(RGB)

    RGB.save("test_image//IMGRGB_%s.jpg" % ( count))


    plt.figure(1)
    plt.imshow(RGB)
    plt.show()















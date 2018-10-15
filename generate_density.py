# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 20:44:01 2018

@author: dell
"""
# 加载的模型库一些东西
# 需要把这个模型加载进更多的图像生成结果
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import scipy
import glob
import os

# 相应的高斯模型结果
Train_image = 267
Crop_W=240
Crop_H=240
Zoom_size=1
def gaussian_filter_density(gt):
    print("gt:", gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    print("density:", density.shape[0])
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    # zip函数是要打包成元组

    pts = list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    #print("pts:", pts)
    leafsize = 2048
    # build kdtree

    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    # query kdtree
    distances, locations = tree.query(pts, k=4)
    #print(distances)
    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            if sigma>5:
                sigma=5



        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant',truncate=2.5)

    print('done.')

    return density


root = 'ShanghaiTech_Crowd_Counting_Dataset/'
print("START!!!")
# now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
path_sets = [part_A_train, part_A_test]
def main():
    for counts in range(Train_image):
        counts = counts + 1
        print("第几个：",counts)
        img = np.array(
            ndimage.imread(os.path.join(part_A_train, "IMG_%s.jpg" % counts), flatten=False))
        # print (img)
        mat = io.loadmat("ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth/GT_IMG_%d.mat" % counts)
        plt.imshow(img)
        # 1表示宽度，0表示高度
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        print(gt[0])
        print(int(gt[0][1]), int(gt[0][0]), img.shape[0])
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        print("k:")
        k = gaussian_filter_density(k)
        """
         下面是采用了一些相应的措施,对于生成的数据加载进入文件中。接下来我们使用npy文件进行保存结果
    
        """
        output_path = "output/train_groundtruth/IMG_output_%d.npy" % counts
        np.save(output_path, k)



    # now see a sample from ShanghaiA

if __name__=="__main__":
    main()

















# 接下来我们进行裁剪的操作过程，初步裁剪成240*240的结果


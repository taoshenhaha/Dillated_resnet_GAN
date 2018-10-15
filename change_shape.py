# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:45:27 2018

@author: dell
"""

import tensorflow as tf

a = tf.constant([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                 [[13,14,15,16],[17,18,19,20],[21,22,23,24]],
                 [[25,26,27,28],[29,30,31,32],[33,34,35,36]]
                 ],
                [[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                 [[13,14,15,16],[17,18,19,20],[21,22,23,24]],
                 [[25,26,27,28],[29,30,31,32],[33,34,35,36]]
                 ]
])



print(a)
values=[]
for i in tf.split(1,3,a):
    
    for ii in tf.split(2,3,i):
        iii=tf.reshape(ii,[2,2,2,1])
        
        values.append(iii)
        ccc=tf.concat(2,values)
    

result_1=tf.split(2,3,ccc) 

sss=tf.concat(1,result_1)

print("自己写的算法，",sss)


#用人家的算法
aaa=tf.depth_to_space(a,2)


with tf.Session() as sess:
    print(sess.run(sss))
    print("系统自带的")
    print(sess.run(aaa))
    
    
    

    
    

   

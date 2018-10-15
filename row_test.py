# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:35:53 2018

@author: dell
"""

import tensorflow as tf
tf.reset_default_graph()
label_path='IMG_output_1_0.npy'
#label_path=tf.convert_to_tensor(label_path)
#input_queue = tf.train.string_input_producer([label_path], shuffle=False)
#input_queue=tf.convert_to_tensor(input_queue)
#reader = tf.FixedLengthRecordReader(record_bytes=240*240*1) 
#key,label = reader.read(label_path) 
label=tf.read_file(label_path)
record_bytes = tf.decode_raw(label, tf.float64)

depth_major = tf.reshape(record_bytes,                       
                        [1, 240, 240])#depth, height, width
print("done")
uint8image = tf.transpose(depth_major, [1, 2, 0])
    
label = tf.cast(uint8image, tf.float32)
with tf.Session() as sess:
    c=sess.run(label)
    print( c)
    

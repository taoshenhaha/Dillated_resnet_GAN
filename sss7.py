import glob
import tensorflow as tf
import matplotlib.pyplot as plt
 
path_list = 'D:/Deeplearning_Demo/image'
img_path = glob.glob(path_list+'*.jpg')
img_path = tf.convert_to_tensor(img_path,dtype=tf.string)
 
# 这里img_path,不放在数组里面
# num_epochs = 1,表示将文件下所有的图片都使用一次
# num_epochs和tf.train.slice_input_producer()中是一样的
# 此参数可以用来设置训练的 epochs
image = tf.train.slice_input_producer([img_path])
 
 
# load one image and decode img
def load_img(path_queue):
    # 创建一个队列读取器，然后解码成数组
    #reader = tf.WholeFileReader()
    #key,file_contents = reader.read(path_queue)
    file_contents = tf.read_file(path_queue[0])
    img = tf.image.decode_jpeg(file_contents,channels=3)
	# 这里很有必要，否则会出错
	# 感觉这个地方貌似只能解码3通道以上的图片
    img = tf.image.resize_images(img,size=(240,240))
    # img = tf.reshape(img,shape=(50,50,4))
    return img
   
img = load_img(image)

image_batch = tf.train.batch([img],batch_size=20)
 
with tf.Session() as sess:
    
    # initializer for num_epochs
    sess.run(tf.local_variables_initializer() )
    sess.run(tf.global_variables_initializer() ) 
    #tf.local_variables_initializer().run() 
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess,coord=coord)
    try:
        while not coord.should_stop():
            imgs = sess.run(image_batch)
            print(imgs.shape)
            print("a")
    except tf.errors.OutOfRangeError:
        print('done')
    
    coord.request_stop()
    coord.join(thread)
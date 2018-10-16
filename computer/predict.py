import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


class_num=3#分类数
image_size=64
num_channels=3


sess = tf.Session()
saver = tf.train.import_meta_graph('./model/car.ckpt-1974.meta')
saver.restore(sess, './model/car.ckpt-1974')
graph = tf.get_default_graph()
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_pred = graph.get_tensor_by_name("y_pred:0")
    
    
    
def predict(x_batch):
    y_test_images = np.zeros((1, class_num)) 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    print(result.argmax())



for text_name in range(1,31):
    images = []
    filename = './model_test/t'+str(text_name)+'.jpg'
    print(filename)
    image = cv2.imread(filename)    
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)    
    images.append(image)    
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0) 
    x_batch = images.reshape(1, image_size,image_size,num_channels)
    predict(x_batch)
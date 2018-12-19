""" 
@author: zoutai
@file: testArgMax.py 
@time: 2018/12/17 
@description: 
"""
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

a = [-7.5781, 7.6791, -4.5542, -2.3470, 2.1828, -2.1568, -9.5705, -6.9970, 3.3440, -3.4133, -5.7267, 5.6748]
ap = tf.argmax(a)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print(sess.run(ap))
# -*- coding: utf-8 -*-
"""
@author: Huhaowen0130
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.layers as layers
import numpy as np
import time

import func

# Load Data
mnist = input_data.read_data_sets('mnist/')
x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels
print('----------%d training examples, %d testing examples' % (x_train.shape[0], x_test.shape[0]))
print('----------image size: {}'.format(x_train[0].shape))

# Hyper-parameters
k = 1000
epochs = 1000
b_size = 128
l_rate = 0.01

# RBF Process
rbf_time = time.time()
centroids, stds = func.rbf_units(x_train, k)
x_train = func.rbf(centroids, stds, x_train)
x_test = func.rbf(centroids, stds, x_test)
print('----------rbf process time = {:.3f} min'.format((time.time() - rbf_time) / 60))

x = tf.placeholder(tf.float32, [None, k])
y = tf.placeholder(tf.int32, [None])
y_one_hot = tf.one_hot(y, 10)
out = layers.fully_connected(x, 10, activation_fn=tf.nn.softmax)

loss = tf.losses.mean_squared_error(y_one_hot, out)
optimizer = tf.train.AdamOptimizer(l_rate)
train_opr = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_one_hot, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

b_num = int(mnist.train.images.shape[0] / b_size)
best_train_acc = 0
best_test_acc = 0
total_time = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epochs):
        start = time.time()
        
        for j in range(b_num):
            b_x = x_train[b_size * j:b_size * j + b_size, :]
            b_y = y_train[b_size * j:b_size * j + b_size]
            sess.run(train_opr, feed_dict={x: b_x, y: b_y})         
        train_time = time.time() - start
        total_time += train_time
        
        train_acc = sess.run(acc, feed_dict={x: x_train, y: y_train})
        test_acc = sess.run(acc, feed_dict={x: x_test, y: y_test})
        best_train_acc = max(best_train_acc, train_acc)
        best_test_acc = max(best_test_acc, test_acc)
        
        print('----------epoch {}/{}: train_acc = {:.3f}, test_acc = {:.3f}, train_time = {:.3f} s'.format(i + 1, epochs, train_acc, test_acc, train_time))
        
    print('----------best_train_acc = {:.3f}, best_test_acc: {:.3f}'.format(best_train_acc, best_test_acc))
    print('----------mean epoch time = {:.3f} s'.format(total_time / epochs))
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import cv2
import time

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 256
IMAGE_PIXELS = IMAGE_WIDTH*IMAGE_HEIGHT
NUM_CLASSES = IMAGE_WIDTH*IMAGE_HEIGHT

def inference(images_placeholder, keep_prob):
	#""" モデルを作成する関数
	#
	#引数: 
	#  images_placeholder: inputs()で作成した画像のplaceholder
	#  keep_prob: dropout率のplace_holder
	#
	#返り値:
	#  cross_entropy: モデルの計算結果
	#"""
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#iremonoを生成
	x  = tf.placeholder(tf.float32, [None,256*300], name = "input")
	y_ = tf.placeholder(tf.float32, [None,256*300], name = "And")
	
	# 画像をリシェイプ 第2引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
	x_image = tf.reshape(x, [-1,256,300,1])


	with tf.name_scope('conv1') as scope:
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	with tf.name_scope('pool1') as scope:
		h_pool1 = max_pool_2x2(h_conv1)

	with tf.name_scope('conv2') as scope:
		W_conv2 = weight_variable([5, 5, 32, 32])
		b_conv2 = bias_variable([32])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	with tf.name_scope('pool2') as scope:
		h_pool2 = max_pool_2x2(h_conv2)

	with tf.name_scope('fc1') as scope:
		W_fc1 = weight_variable([75*64*32, 1024])
		b_fc1 = bias_variable([1024])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 75*64*32])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	with tf.name_scope('fc2') as scope:
		W_fc2 = weight_variable([1024, NUM_CLASSES])
		b_fc2 = bias_variable([NUM_CLASSES])

	with tf.name_scope('sigmoid') as scope:
		y_conv =tf.sigmoid(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

	return y_conv


#######################################################answer making
### training data
train_array = []
train_image = []

### training data label
train_label_array = []
train_label = []

### Read answer image
Ansfiles = os.listdir('train')
for k in range(2000):
	img = cv2.imread("train/"+Ansfiles[k])
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	train_array.append(img.flatten().astype(np.float32)/255.0)

### change to numpy array
train_image = np.asarray(train_array)

images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
keep_prob = tf.placeholder(tf.float32)

logits = inference(images_placeholder, keep_prob)
sess = tf.InteractiveSession()

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
saver.restore(sess, "model.ckpt")

for i in range(len(test_image)):
	pred = tf.sigmoid(logits.eval(feed_dict={x: [test_image[i]],keep_prob: 1.0 })[0])
	print pred#とりあえず出力


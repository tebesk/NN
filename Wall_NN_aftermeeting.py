# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import cv2
import time

#import input_data

# training data
train_array = []
train_image = []
# training data label
train_label_array = []
train_label = []

### Read answer image
Ansfiles = os.listdir('ans')

sess= tf.InteractiveSession()



IMAGE_WIDTH = 300
IMAGE_HEIGHT = 256
IMAGE_PIXELS = IMAGE_WIDTH*IMAGE_HEIGHT
NUM_CLASSES = IMAGE_WIDTH*IMAGE_HEIGHT

#####################################################net
# 重みを標準偏差0.1の正規分布で初期化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# バイアスを標準偏差0.1の正規分布で初期化
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 第一層畳み込み層の作成
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')# 真ん中2つが縦横のストライド

# X2プーリング層の作成
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')# 真ん中2つが縦横のストライド

# X4プーリング層の作成
def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
  
# X4プーリング層の作成
def max_pool_4x128(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 1, 1],strides=[1, 4, 1, 1], padding='VALID')

#グラフを生成
x  = tf.placeholder(tf.float32, [None,IMAGE_PIXELS], name = "input")
y_ = tf.placeholder(tf.float32, [None,IMAGE_PIXELS], name = "And")



### 1層目 畳み込み層
print "before 1st layer"
#the first two dimensions are the patch size, 
#the next is the number of input channels, 
#and the last is the number of output channels. 
#[filter_height , filter_width , in_channels, output_channels]
# 5x5フィルタで32チャネルを出力（入力は白黒画像なので1チャンネル）

# 画像をリシェイプ 第2引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
x_image = tf.reshape(x, [-1,256,300,1])

with tf.name_scope('conv1') as scope:
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

print "before 2nd layer"
### 2層目 プーリング層
# 2x2のマックスプーリング層を構築
with tf.name_scope('pool1') as scope:
	h_pool1 = max_pool_2x2(h_conv1)

### 3層目 畳み込み層
with tf.name_scope('conv2') as scope:
	W_conv2 = weight_variable([5, 5, 32, 32])
	b_conv2 = bias_variable([32])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

### 4層目 プーリング層
with tf.name_scope('pool2') as scope:
	h_pool2 = max_pool_2x2(h_conv2)

print "before 5th layer"
### 5層目 全結合層
# オリジナル画像が256x300で、今回畳み込みでpadding='SAME'を指定しているため
# プーリングでのみ画像サイズが変わる。2x2プーリングで2x2でストライドも2x2なので
# 縦横ともに各層で半減する。そのため、300or256 / 2 / 2 = 75or64が現在の画像サイズ
# 全結合層にするために、1階テンソルに変形。画像サイズ縦と画像サイズ横とチャネル数の積の次元
# 出力は1024（この辺は決めです）　
with tf.name_scope('fc1') as scope:
	W_fc1 = weight_variable([75*64*32, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 75*64*32])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	#Dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

print "befor readout"
#Readout Layer
with tf.name_scope('fc2') as scope:
	W_fc2 = weight_variable([1024, NUM_CLASSES])
	b_fc2 = bias_variable([NUM_CLASSES])
	
with tf.name_scope('sigmoid') as scope:
	y_conv =tf.sigmoid(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

#Train and Evaluate the Model############################################################

# 評価系の関数を用意
cross_entropy = -tf.reduce_sum(tf.square(y_ - y_conv))#-tf.reduce_sum(y_*tf.log(y_conv))←うまく行かず
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

divide = tf.constant([0.7])
correct_prediction = tf.equal(tf.floordiv(y_conv, divide), y_ )#01に修正divideでわって切り捨て
accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"), name = "accuracy")
sess.run(tf.initialize_all_variables())

#tensorboard 書き出し用
tf.scalar_summary("accuracy" , accuracy)

print "befor merge all"
# 全ての要約をマージしてそれらを /tmp/mnist_logs に書き出します。
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/logs",sess.graph)




print "NetWorkMaking Fin"
#実際の計算　一個ずつファイルを読んでいくことにする
st = time.time()

#画像を配列の中にぶっこんでいく！
for k in range(2000):
	#img read(A)
	img = cv2.imread("ans/"+Ansfiles[k])
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
				 
	#img read(T)
	if  os.path.isfile("train/"+Ansfiles[k]):#because Ansfiles are default images(not ans image)
		def_img = cv2.imread("train/"+Ansfiles[k])
		def_img = cv2.cvtColor(def_img, cv2.COLOR_RGB2GRAY)			

		#we put all image in train array
		train_label_array.append(img.flatten().astype(np.float32)/255.0)
		train_array.append(def_img.flatten().astype(np.float32)/255.0)
	else:
		print Ansfiles[k]
		continue

# change to numpy array
train_image = np.asarray(train_array)
train_label = np.asarray(train_label_array)
print train_image.shape
print train_label.shape





# Let's train 
for i in range(2000):
	
	for k in range(50):
		raw_data = train_image[30*k:30*k+30,:]
		ans_data = train_label[30*k:30*k+30,:]
		#print raw_data.shape
		#print ans_data.shape
		##本当の学習
		train_step.run(feed_dict={x:raw_data, y_: ans_data, keep_prob: 0.1})
		if k==49:
			##Evaluating Our Model
			train_accuracy = accuracy.eval({x:raw_data, y_: ans_data, keep_prob: 0.1})
			summary_str= sess.run(merged, feed_dict={x:raw_data, y_: ans_data, keep_prob: 0.1})
			print("step %d, training accuracy %g"%(i, train_accuracy))
			print "***elapsed time %f [s]" % (time.time() - st)
			writer.add_summary(summary_str,i)


#結果の保存
saver = tf.train.Saver()
save_path= saver.save(sess,"/home/ys/Undeux/model.ckpt")
print("model saved")

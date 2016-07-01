# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

# training data
train_array = []
train_image = []
# training data label
train_label_array = []
train_label = []

### Read answer image
Ansfiles = os.listdir('half_ans')
Ansfiles.sort()

### Read input image
#Testfiles = os.listdir('half_img')
#Testfiles.sort()


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
x  = tf.placeholder(tf.float32, shape=[256*300],name = "input")
y_ = tf.placeholder(tf.float32, shape=[256*300], name = "And")

sess= tf.InteractiveSession()

### 1層目 畳み込み層
print "before 1st layer"
#[filter_height , filter_width , in_channels, output_channels]
# 5x5フィルタで32チャネルを出力（入力は白黒画像なので1チャンネル）
W_conv1 = weight_variable([5,5,1,32])
# 畳み込み層のバイアス
b_conv1 = bias_variable([32])
# 画像をリシェイプ 第2引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
x_image = tf.reshape(x, [-1,256,300,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) )

print "before 2nd layer"
### 2層目 プーリング層
# 2x2のマックスプーリング層を構築
h_pool1 = max_pool_2x2(h_conv1)

### 3層目 畳み込み層
W_conv2 = weight_variable([5,5,32,32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2 )

### 4層目 プーリング層
h_pool2 = max_pool_2x2(h_conv2)

print "before 5th layer"
### 5層目 全結合層
# オリジナル画像が512x600で、今回畳み込みでpadding='SAME'を指定しているため
# プーリングでのみ画像サイズが変わる。2x2プーリングで2x2でストライドも2x2なので
# 縦横ともに各層で半減する。そのため、600or512 / 2 / 2 = 150or128が現在の画像サイズ
# 全結合層にするために、1階テンソルに変形。画像サイズ縦と画像サイズ横とチャネル数の積の次元
# 出力は1024（この辺は決めです）　
W_fc1 = weight_variable([75*64*32,1024])
b_fc1 = bias_variable([1024])
print "before poor flat"
h_pool2_flat = tf.reshape(h_pool1,[-1,75*64*32])
print "befor sigmoid"
h_fc1= tf.sigmoid(tf.matmul(h_pool2_flat, W_fc1))
print "befor dropout"

#Dropout
#dropout率を入れるための仮のTensor
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

print "befor readout"
#Readout Layer
W_fc2 = weight_variable([1024,256*300])
b_fc2 = bias_variable([256*300])

print "befor y_CNV"
#y_conv =tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2)+b_fc2) #結論:Sigmoid is better?
y_conv =tf.sigmoid(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

print "befor tf.name_scope"
#Train and Evaluate the Model
with tf.name_scope("difference") as scope:
	difference  = tf.reduce_sum(tf.square(y_ - y_conv))
	#クロスエントロピーを出力
	ce_summ = tf.scalar_summary("difference", difference)
	
	print "befor scope train"
with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(difference)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)): meanは値が微妙なので却下.誤差を512*600で割ってる？

print "befor merge all"
# 全ての要約をマージしてそれらを /tmp/mnist_logs に書き出します。
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/logs",sess.graph)
tf.initialize_all_variables().run()#sess.run(tf.initialize_all_variables())

print "NetWorkMaking Fin"
#実際の計算　一個ずつファイルを読んでいくことにする
st = time.time()

for i in range(100000):
	cost = 0
	summary_str= 0
	for k in range(1000):
	#if i%2 == 0:
		#img read(A)
		img = cv2.imread("half_ans/"+Ansfiles[k])
		img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		#change to 1 line 
		train_label_array = img.flatten().astype(np.float32)/255.0
		#train_label_array.append(img)
		
		#img read(T)
		if  os.path.isfile("half_img/"+Ansfiles[k]):#because Ansfiles are default images(not ans image)
			def_img = cv2.imread("half_img/"+Ansfiles[k])
			def_img = cv2.cvtColor(def_img, cv2.COLOR_RGB2GRAY)
			#change to 1 line 
			train_array = def_img.flatten().astype(np.float32)/255.0
			#train_array.append(def_img)
		else:
			continue

		# change to numpy array
		train_image = np.asarray(train_array)
		train_label = np.asarray(train_label_array)
		

		#data input in NN
		feed = {x:train_image, y_: train_label, keep_prob: 1.0}
		#calculate 
		result=sess.run([merged, difference], feed_dict=feed)
		#summary_str+= result[0]
		cost += result[1]
		
		print str(k) +u":"+str(result[1])

		
		print "***short step time %f [s]" % (time.time() - st)
		train_step.run(feed_dict={x: train_image, y_: train_label, keep_prob: 1.0})

	writer.add_summary(i,difference)
	print("difference at step %s : %s"%(i,cost))
	print "***elapsed time %f [s]" % (time.time() - st)
#print("test accuracy %g"%difference.eval(feed_dict={x:train_image, y_: train_label, keep_prob: 1.0}))

#結果の保存
saver = tf.train.Saver()
save_path= saver.save(sess,"/home/terumo/py/model.ckpt")
print("model saved")


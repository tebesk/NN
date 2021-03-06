# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
# training data
train_image = []
# training data label
train_label = []

# Read answer image
Ansfiles = os.listdir('Ans')
Ansfiles.sort()

# read input image
Testfiles = os.listdir('BD')
Testfiles.sort()

for i in Testfiles:
	#img read
	img = cv2.imread("BD/"+i)
	img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	#1 line 
	img = img.flatten().astype(np.float32)/255.0
	train_image.append(img)
	
for j in Ansfiles:
	img = cv2.imread("Ans/"+j)
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	img = img.flatten().astype(np.float32)/255.0
	train_label.append(img)

# change to numpy array

train_image = np.asarray(train_image)
train_label = np.asarray(train_label)

print train_image.shape
print train_label.shape


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
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# X2プーリング層の作成
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# X4プーリング層の作成
def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
  
# X4プーリング層の作成
def max_pool_4x128(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 1, 1],strides=[1, 4, 1, 1], padding='VALID')

#グラフを生成

x  = tf.placeholder(tf.float32, shape=[None, 308*308],name = "input")
y_ = tf.placeholder(tf.float32, shape=[None, 308*308], name = "And")

sess= tf.InteractiveSession()

#1st layer

#[filter_height , filter_width , in_channels, output_channels]
W_conv1 = weight_variable([5,5,1,32])

b_conv1 = bias_variable([32])
# 画像をリシェイプ 第2引数は画像数(-1は元サイズを保存するように自動計算)、縦x横、チャネル
x_image = tf.reshape(x, [-1,308,308,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) )
h_pool1 = max_pool_2x2(h_conv1)

#2nd layer
W_conv2 = weight_variable([5,5,32,32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2 )
h_pool2 = max_pool_2x2(h_conv2)

### 5層目 全結合層
# オリジナル画像が308x308で、今回畳み込みでpadding='SAME'を指定しているため
# プーリングでのみ画像サイズが変わる。2x2プーリングで2x2でストライドも2x2なので
# 縦横ともに各層で半減する。そのため、308 / 2 / 2 = 77が現在の画像サイズ
# 全結合層にするために、1階テンソルに変形。画像サイズ縦と画像サイズ横とチャネル数の積の次元
# 出力は1024（この辺は決めです）　あとはSoftmax Regressionと同じ
W_fc1 = weight_variable([77*77*32,1024*2])
b_fc1 = bias_variable([1024*2])

h_pool2_flat = tf.reshape(h_pool2,[-1,77*77*32])
h_fc1= tf.sigmoid(tf.matmul(h_pool2_flat, W_fc1)+b_fc1 )

#Dropout
#dropout率を入れるための仮のTensor
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = weight_variable([1024*2,308*308])
b_fc2 = bias_variable([308*308])
#y_conv =tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)#結論:Sigmoid is better?
y_conv =tf.sigmoid(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)#結論


#Train and Evaluate the Model
with tf.name_scope("difference") as scope:
	difference  = tf.reduce_sum(tf.square(y_ - y_conv))
	#クロスエントロピーを出力
	ce_summ = tf.scalar_summary("difference", difference)
	
with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(difference)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)): meanは値が微妙なので却下誤差を308*308で割ってる？

batch_xs = train_image
batch_ys = train_label
print batch_xs.shape
print batch_ys.shape

# 全ての要約をマージしてそれらを /tmp/mnist_logs に書き出します。
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/logs",sess.graph)
tf.initialize_all_variables().run()#sess.run(tf.initialize_all_variables())

print "NetWorkMaking Fin"

#実際の計算
st = time.time()

for i in range(10):
	
	if i%2 == 0:
		feed = {x:batch_xs, y_: batch_ys, keep_prob: 1.0}
		result=sess.run([merged,difference], feed_dict=feed)
		summary_str=result[0]
		acc=result[1]
		writer.add_summary(summary_str,i)
		print("difference at step %s : %s"%(i,acc))
		#何故か以下のコードだとGraphが動かない
		#train_accuracy = difference.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
		#print("step %d, training accuracy %g"%(i, train_accuracy))
		#summary_writer.add_summary(train_accuracy, i)
		print "***elapsed time %f [s]" % (time.time() - st)
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
print("test accuracy %g"%difference.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0}))

#結果の保存
saver = tf.train.Saver()
save_path= saver.save(sess,"/home/ys/python/model.ckpt")
print("model saved")


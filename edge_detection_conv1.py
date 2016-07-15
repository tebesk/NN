#!/usr/bin/env python
# -*- coding: utf-8 -*-

#http://kivantium.hateblo.jp/entry/2016/02/04/213050
from __future__ import print_function
import cv2
import argparse
import os

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import random
import numpy as np

# クラスの定義
class Conv(chainer.Chain):
    def __init__(self):
        super(Conv, self).__init__(
            # 入力・出力1ch, ksize=3
            conv1=L.Convolution2D(1, 1, 5, stride=1, pad=2),
        )

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, x):
        self.clear()
        h = self.conv1(x)
        return h

    def calc_loss(self, x, t):
        self.clear()
        h = self.conv1(x)
        loss = F.mean_squared_error(h, t)
        return loss 

### Read answer image
Ansfiles = os.listdir('ans_area')

# 学習対象のモデル作成
model = Conv()
train_image = chainer.Variable(np.asarray([[cv2.imread("ans/"+random.choice(Ansfiles),0)/255.0]], dtype=np.float32))

# 最適化の設定
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習

for filename in range(100):
	train_image = chainer.Variable(np.asarray([[cv2.imread("denoised/"+Ansfiles[filename], 0)/255.0]], dtype=np.float32))
	target = chainer.Variable(np.asarray([[cv2.imread("ans/"+Ansfiles[filename], 0)/255.0]], dtype=np.float32))
	
	for seq in range(10000):
#	for filename in random.sample(Ansfiles,20):
#		train_image = chainer.Variable(np.asarray([[cv2.imread("denoised/"+filename, 0)/255.0]], dtype=np.float32))
#		target = chainer.Variable(np.asarray([[cv2.imread("ans/"+filename, 0)/255.0]], dtype=np.float32))
		loss = model.calc_loss(train_image, target)	
		model.zerograds()
		loss.backward()
		optimizer.update()
		if seq%200==0:
			print("{}: {}".format(seq, loss.data))
	print(model.conv1.W.data[0][0])
	trained = model.forward(train_image).data[0][0]*255
	cv2.imwrite("trained.jpg", trained)


# 学習結果の表示
print(model.conv1.W.data[0][0])
for filename in Ansfiles:
	train_image = chainer.Variable(np.asarray([[cv2.imread("denoised/"+filename, 0)/255.0]], dtype=np.float32))
	trained = model.forward(train_image).data[0][0]*255
	cv2.imwrite("_trained/"+filename, trained)
	
	
#
#  実験プロセス
#  1つのファイルを１００００ステップ分学習させ、あえて徹底的にオーバーフィッティングさせる
#  そのモデルで次のファイルを学習させる 
#  これを１００ファイル分だけ実施
#
#  エッジだけのものを答えにした結果。 このファイルで実験したときの  
# １００ファイル目　9800Stepで: mean square error は0.00471516372636　となっている
#[[-0.09336051  0.1853829   0.04092094 -0.14387129  0.09618491]
# [ 0.02177362 -0.05224365 -0.09952049  0.00280143  0.07434209]
# [ 0.13707821 -0.30128592 -0.08413173  0.18352953  0.03772697]
# [-0.02080819 -0.12694013  0.00848041  0.01496854  0.11795343]
# [-0.15959556  0.23649965  0.0546778  -0.03958907 -0.02575691]]
#


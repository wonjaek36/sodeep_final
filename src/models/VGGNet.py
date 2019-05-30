import tensorflow as tf
from tensorflow import keras
import numpy as np


class VGGNet(tf.keras.Model):

	def __init__(self):
		self.config = config

	def create_placeholders(self, height, width, channel, output):
		data_type = self.config['MODEL']['data']
		width = self.config['DATA'][data_type]['width']
		height = self.config['DATA'][data_type]['height']
		channel = self.config['DATA'][data_type]['channel']
		output = self.config['DATA'][data_type]['output']

		self.width = width
		self.height = height
		self.channel = channel
		self.output = output

		X = tf.placeholder(tf.float32, shape=(None, height, width, channel))
		Y = tf.placeholder(tf.float32, shape=(None, output))

		return X, Y

	def forward_propagation(self, X, parameters, dropout=0):

		width = self.width
		height = self.height
		channel = self.channel

		model = tf.keras.Sequential([
			# VGG 1 layer
			tf.keras.layers.Conv2D(filter=64, kernel_size=(3,3), strides=(1,1), input_shape=(width, height, channel), padding='same', name='Z1_1', activation='relu'),
			tf.keras.layers.Conv2D(filter=64, kernel_size=(3,3), strides=(1,1), name='Z1_2', padding='same', activation='relu', name='Z1_2'),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG 2 layer
			tf.keras.layers.Conv2D(filter=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z2_1'),
			tf.keras.layers.Conv2D(filter=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z2_2'),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG 3 layer
			tf.keras.layers.Conv2D(filter=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
			tf.keras.layers.Conv2D(filter=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG 4 layer
			tf.keras.layers.Conv2D(filter=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
			tf.keras.layers.Conv2D(filter=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

			#VGG 5 layer
			tf.keras.layers.Conv2D(filter=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
			tf.keras.layers.Conv2D(filter=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

			#VGG FC
			tf.keras.layers.Flatten()
			tf.keras.layers.Dense(4096, activation='relu', name='fc1')
		])
		W1_1 = parameters['W1_1']
		W1_2 = parameters['W1_2']
		W2_1 = parameters['W2_1']
		W2_2 = parameters['W2_2']
		W3_1 = parameters['W3_1']
		W3_2 = parameters['W3_2']
		W3_3 = parameters['W3_3']
		W4_1 = parameters['W4_1']
		W4_2 = parameters['W4_2']
		W4_3 = parameters['W4_3']
		W5_1 = parameters['W5_1']
		W5_2 = parameters['W5_2']
		W5_3 = parameters['W5_3']
		W6 = parameters['W6']
		W7 = parameters['W7']
		W8 = parameters['W8']

		# VGG 1 layer
		stride = 3
		Z1_1 = tf.nn.conv2d(X, W1_1, strides=[1, stride, stride, 1], padding='SAME')
		Z1_2 = tf.nn.conv2d(Z1_1, W1_2, strides=[1, stride, stride, 1], padding='SAME')
		A1 = tf.nn.relu(Z1_2)
		
		stride = 2
		filtr = 2
		A1 = tf.nn.max_pool(A1, ksize=[1, filtr, filtr, 1], strides=[1, stride, stride, 1], padding='SAME')

		# VGG 2 layer
		stride = 3
		Z2_1 = tf.nn.conv2d(A1, W2_1, strides=[1, stride, stride, 1], padding='SAME')
		Z2_2 = tf.nn.conv2d(Z2_1, W2_2, strides=[1, stride, stride, 1], padding='SAME')
		A2 = tf.nn.relu(Z2_2)
		
		stride = 2
		filtr = 2
		A2 = tf.nn.max_pool(A2, ksize=[1, filtr, filtr, 1], strides=[1, stride, stride, 1], padding='SAME')

		# VGG 3 layer
		stride = 3
		Z3_1 = tf.nn.conv2d(A2, W3_1, strides=[1, stride, stride, 1], padding='SAME')
		Z3_2 = tf.nn.conv2d(Z3_1, W3_2, strides=[1, stride, stride, 1], padding='SAME')
		Z3_3 = tf.nn.conv2d(Z3_2, W3_3, strides=[1, stride, stride, 1], padding='SAME')
		A3 = tf.nn.relu(Z3_3)
		
		filtr = 2
		stride = 2
		A3 = tf.nn.max_pool(A3, ksize=[1, filtr, filtr, 1], strides=[1, stride, stride,1], padding='SAME')

		# VGG 4 layer
		stride = 3
		Z4_1 = tf.nn.conv2d(A3, W4_1, strides=[1, stride, stride, 1], padding='SAME')
		Z4_2 = tf.nn.conv2d(Z4_1, W4_2, strides=[1, stride, stride, 1], padding='SAME')
		Z4_3 = tf.nn.conv2d(Z4_2, W4_3, strides=[1, stride, stride, 1], padding='SAME')
		A4 = tf.nn.relu(Z4_3)

		filtr = 2
		stride = 2
		A4 = tf.nn.max_pool(A4, ksize=[1, filtr, filtr, 1], strides=[1, stride, stride, 1], padding='SAME')

		# VGG 5 layer
		stride = 3
		Z5_1 = tf.nn.conv2d(A4, W5_1, strides=[1, stride, stride, 1], padding='SAME')
		Z5_2 = tf.nn.conv2d(Z5_1, W5_2, strides=[1, stride, stride, 1], padding='SAME')
		Z5_3 = tf.nn.conv2d(Z5_2, W5_3, strides=[1, stride, stride, 1], padding='SAME')
		A5 = tf.nn.relu(Z5_3)

		filtr = 2
		stride = 2
		A5 = tf.nn.max_pool(A5, ksize=[1, filtr, filtr, 1], strides=[1, stride, stride, 1], padding='SAME')

		# VGG FC
		FC1 = tf.contrib.layers.flatten(A5)
		#TODO add dropout

		FC2 = tf.nn.relu(tf.matmul(FC1, W6))
		#TODO add dropout

		FC3 = tf.nn.relu(tf.matmul(FC2, W7))
		#TODO add dropout

		Z = tf.nn.relu(tf.matmul(FC3, W8)) # before softmax

		return Z








		
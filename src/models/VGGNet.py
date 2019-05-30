import tensorflow as tf
from tensorflow import keras
import numpy as np


class VGGNet:

	def __init__(self, config):
		self.config = config
		self.data_type = config['MODEL']['data']
		data_type = self.data_type
		self.width = config['DATA'][data_type]['width']
		self.height = config['DATA'][data_type]['height']
		self.channel = config['DATA'][data_type]['channel']
		self.output = config['DATA'][data_type]['output']

	def create_placeholders(self, height, width, channel, output):

		self.width = width
		self.height = height
		self.channel = channel
		self.output = output

		X = tf.placeholder(tf.float32, shape=(None, height, width, channel))
		Y = tf.placeholder(tf.float32, shape=(None, output))

		return X, Y

	def create_keras_model(self, dropout=0):

		width = self.width
		height = self.height
		channel = self.channel
		output = self.output
		zero_padding = 3

		conv2d_final_width = (width+2*zero_padding) // 2 // 2 // 2 // 2 // 2
		conv2d_final_height = (height+2*zero_padding) // 2 // 2 // 2 // 2 // 2
		flatten_neuron = conv2d_final_height*conv2d_final_width*512

		print (conv2d_final_width)
		# TODO
		# Add kernel initializer "glorot_uniform"
		model = tf.keras.Sequential([
			# Zero padding 3
			tf.keras.layers.ZeroPadding2D(padding=(zero_padding, zero_padding), input_shape=(width, height, channel)),

			# VGG 1 layer
			tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='Z1_1', activation='relu'),
			tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),  padding='same', activation='relu', name='Z1_2'),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG 2 layer
			tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z2_1'),
			tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z2_2'),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG 3 layer
			tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z3_1'),
			tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z3_2'),
			tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z3_3'),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG 4 layer
			tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z4_1'),
			tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z4_2'),
			tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z4_3'),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

			#VGG 5 layer
			tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z5_1'),
			tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z5_2'),
			tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z5_3'),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

			#VGG FC
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(flatten_neuron, activation='relu', name='fc1'),
			tf.keras.layers.Dense(flatten_neuron, activation='relu', name='fc2'),
			tf.keras.layers.Dense(output, activation='softmax', name='fc3')
		])
		
		return model








		
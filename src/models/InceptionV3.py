import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import json
import sys


class InceptionV3():

    def __init__(self, config):
        self.config = config
        self.data_type = config['MODEL']['data']
        data_type = self.data_type
        self.width = config['DATA'][data_type]['width']
        self.height = config['DATA'][data_type]['height']
        self.channel = config['DATA'][data_type]['channel']
        self.output = config['DATA'][data_type]['output']

    def conv2d_bn(self, X, filters, kernel_size, padding='same', strides=(1,1), name=None):

        if name is not None:
            conv2d_name = name + '_conv2d'
            bn_name = name + '_batchNorm'
            act_name = name + '_activation'

        else:
            conv2d_name = None
            bn_name = None
            act_name = None

        X = tf.keras.layers.Conv2D(filters = filters, kernel_size=kernel_size, strides=strides, padding=padding, name=conv2d_name)(X)
        X = tf.keras.layers.BatchNormalization(axis=3, scale=False, name=bn_name)(X)
        X = tf.keras.layers.Activation('relu', name=act_name)(X)

        return X


    def create_keras_model(self, options={}):
        config = self.config

        data_type = config['MODEL']['data']
        width = self.width
        height = self.height
        channel = self.channel
        output = self.output

        inputs = tf.keras.layers.Input(shape=(width, height, channel)) # 32x32x3

        if data_type == 'base' or data_type == 'cifar10' or data_type == 'cifar100':
            X = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(inputs)
        else:
            X = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inputs)

        # First
        if data_type == 'base' or data_type == 'cifar10' or data_type == 'cifar100':
            X = self.conv2d_bn(X, filters=32, kernel_size=(3,3), padding='same', name='s1')
        else:
            X = self.conv2d_bn(X, filters=32, kernel_size=(3,3), padding='valid', name='s1')

        X = self.conv2d_bn(X, filters=64, kernel_size=(3,3), name='s2')

        if data_type == 'base' or data_type == 'cifar10' or data_type == 'cifar100':
            X = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1))(X)
        else:
            X = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(X)
        

        # Second
        X = self.conv2d_bn(X, filters=80, kernel_size=(1, 1), padding='valid', name='s3')
        if data_type == 'base' or data_type == 'cifar10' or data_type == 'cifar100':
            X = self.conv2d_bn(X, filters=192, kernel_size=(3, 3), padding='same', name='s4')
        else:
            X = self.conv2d_bn(X, filters=192, kernel_size=(3, 3), padding='valid', name='s4')
        
        if data_type == 'base' or data_type == 'cifar10' or data_type == 'cifar100':
            X = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1))(X)
        else:
            X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Mixed 0 288 (/4, /4)
        branch33d = self.conv2d_bn(X, filters=64, kernel_size=(1, 1))
        branch33d = self.conv2d_bn(branch33d, filters=96, kernel_size=(3, 3))
        branch33d = self.conv2d_bn(branch33d, filters=96, kernel_size=(3, 3))
        
        branch33 = self.conv2d_bn(X, filters=48, kernel_size=(1, 1))
        branch33 = self.conv2d_bn(branch33, filters=64, kernel_size=(5,5))
        
        branch11 = self.conv2d_bn(X, filters=64, kernel_size=(1,1))
        
        branchap = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(X)
        branchap = self.conv2d_bn(branchap, filters=64, kernel_size=(1, 1))

        X = tf.keras.layers.Concatenate(axis=3)([branch11, branch33d, branch33, branchap])

        # Mixed 1 288, (/4, /4)
        branch33d = self.conv2d_bn(X, filters=64, kernel_size=(1, 1))
        branch33d = self.conv2d_bn(branch33d, filters=96, kernel_size=(3, 3))
        branch33d = self.conv2d_bn(branch33d, filters=96, kernel_size=(3, 3))
        
        branch33 = self.conv2d_bn(X, filters=48, kernel_size=(1, 1))
        branch33 = self.conv2d_bn(branch33, filters=64, kernel_size=(5,5))
        
        branch11 = self.conv2d_bn(X, filters=64, kernel_size=(1,1))
        
        branchap = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(X)
        branchap = self.conv2d_bn(branchap, filters=64, kernel_size=(1, 1))

        X = tf.keras.layers.Concatenate(axis=3)([branch11, branch33d, branch33, branchap])

        # Mixed 2 288, (/4, /4)
        branch33d = self.conv2d_bn(X, filters=64, kernel_size=(1, 1))
        branch33d = self.conv2d_bn(branch33d, filters=96, kernel_size=(3, 3))
        branch33d = self.conv2d_bn(branch33d, filters=96, kernel_size=(3, 3))
        
        branch33 = self.conv2d_bn(X, filters=48, kernel_size=(1, 1))
        branch33 = self.conv2d_bn(branch33, filters=64, kernel_size=(5,5))
        
        branch11 = self.conv2d_bn(X, filters=64, kernel_size=(1,1))
        
        branchap = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(X)
        branchap = self.conv2d_bn(branchap, filters=64, kernel_size=(1, 1))

        X = tf.keras.layers.Concatenate(axis=3)([branch11, branch33d, branch33, branchap])

        # Mixed 3, 768, (/8, /8)
        branch33 = self.conv2d_bn(X, filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid')

        branch33d = self.conv2d_bn(X, filters=64, kernel_size=(1, 1))
        branch33d = self.conv2d_bn(branch33d, filters=96, kernel_size=(3, 3))
        branch33d = self.conv2d_bn(branch33d, filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid')

        branchmp = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(X)
        X = tf.keras.layers.Concatenate(axis=3)([branch33, branch33d, branchmp])

        
        # Mixed 4, 768, (/8, /8)
        branch11 = self.conv2d_bn(X, filters=192, kernel_size=(1, 1))

        branch77 = self.conv2d_bn(X, filters=128, kernel_size=(1, 1))
        branch77 = self.conv2d_bn(branch77, filters=128, kernel_size=(1, 7))
        branch77 = self.conv2d_bn(branch77, filters=192, kernel_size=(7, 1))

        branch77d = self.conv2d_bn(X, filters=128, kernel_size=(1, 1))
        branch77d = self.conv2d_bn(branch77d, filters=128, kernel_size=(7, 1))
        branch77d = self.conv2d_bn(branch77d, filters=128, kernel_size=(1, 7))
        branch77d = self.conv2d_bn(branch77d, filters=128, kernel_size=(7, 1))
        branch77d = self.conv2d_bn(branch77d, filters=192, kernel_size=(1, 7))

        branchap = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(X)
        branchap = self.conv2d_bn(branchap, filters=192, kernel_size=(1, 1))
        X = tf.keras.layers.Concatenate(axis=3)([branch11, branch77, branch77d, branchap])


        # Mixed 5, 768, (/8, /8)
        branch11 = self.conv2d_bn(X, filters=192, kernel_size=(1, 1))

        branch77 = self.conv2d_bn(X, filters=160, kernel_size=(1, 1))
        branch77 = self.conv2d_bn(branch77, filters=160, kernel_size=(1, 7))
        branch77 = self.conv2d_bn(branch77, filters=192, kernel_size=(7, 1))

        branch77d = self.conv2d_bn(X, filters=160, kernel_size=(1, 1))
        branch77d = self.conv2d_bn(branch77d, filters=160, kernel_size=(7, 1))
        branch77d = self.conv2d_bn(branch77d, filters=160, kernel_size=(1, 7))
        branch77d = self.conv2d_bn(branch77d, filters=160, kernel_size=(7, 1))
        branch77d = self.conv2d_bn(branch77d, filters=192, kernel_size=(1, 7))

        branchap = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(X)
        branchap = self.conv2d_bn(branchap, filters=192, kernel_size=(1, 1))
        X = tf.keras.layers.Concatenate(axis=3)([branch11, branch77, branch77d, branchap])
        

        # Mixed 6, 768, (/8, /8)
        branch11 = self.conv2d_bn(X, filters=192, kernel_size=(1, 1))

        branch77 = self.conv2d_bn(X, filters=160, kernel_size=(1, 1))
        branch77 = self.conv2d_bn(branch77, filters=160, kernel_size=(1, 7))
        branch77 = self.conv2d_bn(branch77, filters=192, kernel_size=(7, 1))

        branch77d = self.conv2d_bn(X, filters=160, kernel_size=(1, 1))
        branch77d = self.conv2d_bn(branch77d, filters=160, kernel_size=(7, 1))
        branch77d = self.conv2d_bn(branch77d, filters=160, kernel_size=(1, 7))
        branch77d = self.conv2d_bn(branch77d, filters=160, kernel_size=(7, 1))
        branch77d = self.conv2d_bn(branch77d, filters=192, kernel_size=(1, 7))

        branchap = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(X)
        branchap = self.conv2d_bn(branchap, filters=192, kernel_size=(1, 1))
        X = tf.keras.layers.Concatenate(axis=3)([branch11, branch77, branch77d, branchap])

        # Mixed 7 768, (/8 /8)
        branch11 = self.conv2d_bn(X, filters=192, kernel_size=(1, 1))

        branch77 = self.conv2d_bn(X, filters=192, kernel_size=(1, 1))
        branch77 = self.conv2d_bn(branch77, filters=192, kernel_size=(1, 7))
        branch77 = self.conv2d_bn(branch77, filters=192, kernel_size=(7, 1))

        branch77d = self.conv2d_bn(X, filters=192, kernel_size=(1, 1))
        branch77d = self.conv2d_bn(branch77d, filters=192, kernel_size=(7, 1))
        branch77d = self.conv2d_bn(branch77d, filters=192, kernel_size=(1, 7))
        branch77d = self.conv2d_bn(branch77d, filters=192, kernel_size=(7, 1))
        branch77d = self.conv2d_bn(branch77d, filters=192, kernel_size=(1, 7))

        branchap = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(X)
        branchap = self.conv2d_bn(branchap, filters=192, kernel_size=(1, 1))
        x = tf.keras.layers.Concatenate(axis=3)([branch11, branch77, branch77d, branchap])

        # Mixed 8, 1280, (/16, /16)
        branch33 = self.conv2d_bn(X, filters=192, kernel_size=(1, 1))
        branch33 = self.conv2d_bn(branch33, filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid')

        branch77 = self.conv2d_bn(X, filters=192, kernel_size=(1, 1))
        branch77 = self.conv2d_bn(branch77, filters=192, kernel_size=(1, 7))
        branch77 = self.conv2d_bn(branch77, filters=192, kernel_size=(7, 1))
        branch77 = self.conv2d_bn(branch77, filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid')

        branchmp = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)
        X = tf.keras.layers.Concatenate(axis=3)([branch33, branch77, branchmp])

        # Mixed 9, 2048, (/16, /16)
        branch11 = self.conv2d_bn(X, filters=320, kernel_size=(1, 1))

        branch33 = self.conv2d_bn(X, filters=384, kernel_size=(1, 1))
        branch33_1 = self.conv2d_bn(branch33, filters=384, kernel_size=(1, 3))
        branch33_2 = self.conv2d_bn(branch33, filters=384, kernel_size=(3, 1))
        branch33 = tf.keras.layers.Concatenate(axis=3)([branch33_1, branch33_2])

        branch33d = self.conv2d_bn(X, filters=448, kernel_size=(1, 1))
        branch33d = self.conv2d_bn(branch33d, filters=384, kernel_size=(3, 3))
        branch33d_1 = self.conv2d_bn(branch33d, filters=384, kernel_size=(1, 3))
        branch33d_2 = self.conv2d_bn(branch33d, filters=384, kernel_size=(3, 1))
        branch33d = tf.keras.layers.Concatenate(axis=3)([branch33d_1, branch33d_2])

        branchap = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(X)
        branchap = self.conv2d_bn(branchap, filters=192, kernel_size=(1, 1))
        X = tf.keras.layers.Concatenate(axis=3)([branch11, branch33, branch33d, branchap])

        model = tf.keras.models.Model(inputs=inputs, outputs=X)

        return model






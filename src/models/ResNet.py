import tensorflow as tf
from tensorflow import keras
import numpy as np


# ResNet-50 keras
class ResNet():

    def __init__(self, config):
        self.config = config
        self.data_type = config['MODEL']['data']
        data_type = self.data_type
        self.width = config['DATA'][data_type]['width']
        self.height = config['DATA'][data_type]['height']
        self.channel = config['DATA'][data_type]['channel']
        self.output = config['DATA'][data_type]['output']

    def identity_block(self, X, k, filters, name):

        X_shortcut = X
        F1, F2, F3 = filters

        X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding='valid', name=name+'_1', activation='relu',
            kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)
        
        X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(k, k), strides=(1,1), padding='same', name=name+'_2', activation='relu',
            kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1,1), padding='valid', name=name+'_3', activation='relu',
            kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)

        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        return X

    def convolution_block(self, X, filters, k, name, s=2):

        X_shortcut = X
        F1, F2, F3 = filters
        
        # Shortcut path
        X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), name = name+'_shortcut', padding='valid',
            kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)


        X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s,s), padding='valid', name=name+'_1', activation='relu',
            kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(k, k), strides=(1,1), padding='same', name=name+'_2', activation='relu',
            kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1,1), padding='valid', name=name+'_3', activation='relu',
            kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        return X

    def create_keras_model(self, options={}):
        
        config = self.config

        data_type = config['MODEL']['data']
        width = self.width
        height = self.height
        channel = self.channel
        output = self.output

        # Initialize keras input
        inputs = tf.keras.layers.Input(shape=(width, height, channel))

        # ResNet 1 layer
        if not (data_type == 'base' or data_type == 'cifar10' or data_type == 'cifar100'):
            X = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', name='Z1_1', activation='relu',
                kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(inputs)
        else:
            X = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding='same', name='Z1_1', activation='relu',
                kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(inputs)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)
        if not (data_type == 'base' or data_type == 'cifar10' or data_type == 'cifar100'):
            X = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

        # ResNet 2 layer
        X = self.convolution_block(X, filters=[64, 64, 256], k=3, name='Z2', s=1)
        X = self.identity_block(X, filters=[64, 64, 256], k=3, name='Z3')
        X = self.identity_block(X, filters=[64, 64, 256], k=3, name='Z4')

        # ResNet 3 layer
        X = self.convolution_block(X, filters=[128, 128, 512], k=3, name='Z5', s=2)
        X = self.identity_block(X, filters=[128, 128, 512], k=3, name='Z6')
        X = self.identity_block(X, filters=[128, 128, 512], k=3, name='Z7')
        X = self.identity_block(X, filters=[128, 128, 512], k=3, name='Z8')

        # ResNet 4 layer
        X = self.convolution_block(X, filters=[256, 256, 1024], k=3, name='Z9', s=2)
        X = self.identity_block(X, filters=[256, 256, 1024], k=3, name='Z10')
        X = self.identity_block(X, filters=[256, 256, 1024], k=3, name='Z11')
        X = self.identity_block(X, filters=[256, 256, 1024], k=3, name='Z12')

        # ResNet 5 layer
        X = self.convolution_block(X, filters=[512, 512, 2048], k=3, name='Z13', s=2)
        X = self.identity_block(X, filters=[512, 512, 2048], k=3, name='Z14')
        X = self.identity_block(X, filters=[512, 512, 2048], k=3, name='Z15')
        X = self.identity_block(X, filters=[512, 512, 2048], k=3, name='Z16')        

        X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid', name="avg_pool")(X)

        # ResNet FC
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(output, activation='relu', name='fc1', 
            kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)

        # Softmax
        X = tf.keras.layers.Activation(activation=tf.keras.activations.softmax)(X)
        
        # Finish Model
        model = tf.keras.models.Model(inputs=inputs, outputs=X)

        return model








        
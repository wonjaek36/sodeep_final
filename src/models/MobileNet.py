import tensorflow as tf
from tensorflow import keras
import numpy as np

class MobileNet():

    def __init__(self, config):
        self.config = config
        self.data_type = config['MODEL']['data']
        data_type = self.data_type
        self.width = config['DATA'][data_type]['width']
        self.height = config['DATA'][data_type]['height']
        self.channel = config['DATA'][data_type]['channel']
        self.output = config['DATA'][data_type]['output']

    def create_keras_model(self, options={}):
        config = self.config

        data_type = config['MODEL']['data']
        width = self.width
        height = self.height
        channel = self.channel
        output = self.output

        regularizer1    = None
        regularizer2    = None
        regularizer3    = None
        regularizer4    = None
        regularizer5    = None
        regularizer6    = None
        regularizer7    = None
        regularizer8    = None
        regularizer9    = None
        regularizer10   = None
        regularizer11   = None
        regularizer12   = None
        regularizer13   = None
        regularizer14   = None

        # MobileNet V1
        inputs = tf.keras.layers.Input(shape=(width, height, channel)) # 32x32x3

        # MobileNet 1 layer
        X = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inputs)
        X = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='valid', name='Z1', activation='relu',
            kernel_regularizer=regularizer1, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X) #16x16x32
        
        # MobileNet 2 layer- depthwise and pointwise layer
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', name='Z2_1', activation='relu',
            kernel_regularizer=regularizer2, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same', name='Z2_2', activation='relu',
            kernel_regularizer=regularizer2, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X) # 16x16x64

        # MobileNet 3 layer- depthwise and pointwise layer
        X = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(X)
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='valid', name='Z3_1', activation='relu',
            kernel_regularizer=regularizer3, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)
        
        X = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding='same', name='Z3_2', activation='relu',
            kernel_regularizer=regularizer3, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X) #8x8x64


        # MobileNet 4 layer- depthwise and pointwise layer
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', name='Z4_1', activation='relu',
            kernel_regularizer=regularizer4, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding='same', name='Z4_2', activation='relu',
            kernel_regularizer=regularizer4, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        # MobileNet 5 layer- depthwise and pointwise layer
        X = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(X)
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='valid', name='Z5_1', activation='relu',
            kernel_regularizer=regularizer5, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='same', name='Z5_2', activation='relu',
            kernel_regularizer=regularizer5, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X) #4x4

        # MobileNet 6 layer- depthwise and pointwise layer
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', name='Z6_1', activation='relu',
            kernel_regularizer=regularizer6, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='same', name='Z6_2', activation='relu',
            kernel_regularizer=regularizer6, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)


        # MobileNet 7 layer- depthwise and pointwise layer
        if not (data_type == 'base' or data_type == 'cifar10' or data_type == 'cifar100' or data_type == 'intel'):
            X = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(X)
            X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='valid', name='Z7_1', activation='relu',
                kernel_regularizer=regularizer7, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same', name='Z7_2', activation='relu',
            kernel_regularizer=regularizer7, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        # ---- Repeat five times
        # MobileNet 8 layer- depthwise and pointwise layer
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', name='Z8_1', activation='relu',
            kernel_regularizer=regularizer8, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same', name='Z8_2', activation='relu',
            kernel_regularizer=regularizer8, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        # MobileNet 9 layer- depthwise and pointwise layer
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', name='Z9_1', activation='relu',
            kernel_regularizer=regularizer9, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same', name='Z9_2', activation='relu',
            kernel_regularizer=regularizer9, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        # MobileNet 10 layer- depthwise and pointwise layer
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', name='Z10_1', activation='relu',
            kernel_regularizer=regularizer10, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same', name='Z10_2', activation='relu',
            kernel_regularizer=regularizer10, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        # MobileNet 11 layer- depthwise and pointwise layer
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', name='Z11_1', activation='relu',
            kernel_regularizer=regularizer11, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same', name='Z11_2', activation='relu',
            kernel_regularizer=regularizer11, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        # MobileNet 12 layer- depthwise and pointwise layer
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', name='Z12_1', activation='relu',
            kernel_regularizer=regularizer12, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same', name='Z12_2', activation='relu',
            kernel_regularizer=regularizer12, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)
        
        # ----

        if (data_type == 'base' or data_type == 'cifar10' or data_type == 'cifar100'):
            # MobileNet 13 layer- depthwise and pointwise layer
            X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', name='Z13_1', activation='relu',
                kernel_regularizer=regularizer12, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

            X = tf.keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same', name='Z13_2', activation='relu',
                kernel_regularizer=regularizer12, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        elif data_type == 'intel':
            #MobileNet 13 layer
            X = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(X)
            X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='valid', name='Z13_1', activation='relu',
                kernel_regularizer=regularizer13, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

            X = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=(1,1), padding='same', name='Z13_2', activation='relu',
                kernel_regularizer=regularizer13, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)
            
            #MobileNet 14 layer
            X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', name='Z14_1', activation='relu',
                kernel_regularizer=regularizer14, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

            X = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=(1,1), padding='same', name='Z14_2', activation='relu',
                kernel_regularizer=regularizer14, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)


        else:
            #MobileNet 13 layer
            X = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(X)
            X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='valid', name='Z13_1', activation='relu',
                kernel_regularizer=regularizer13, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

            X = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=(1,1), padding='same', name='Z13_2', activation='relu',
                kernel_regularizer=regularizer13, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)
            
            #MobileNet 14 layer
            X = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', name='Z14_1', activation='relu',
                kernel_regularizer=regularizer14, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

            X = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=(1,1), padding='same', name='Z14_2', activation='relu',
                kernel_regularizer=regularizer14, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(X)

        #MobileNet Average
        X = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(1,1), padding='valid')(X)

        # Fully Connected Layer
        X = tf.keras.layers.Flatten()(X)
        # TODO add dropout
        X = tf.keras.layers.Dense(output, activation='relu', name='fc1', kernel_regularizer=regularizer6,
            kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)

        # Softmax
        X = tf.keras.layers.Activation(activation=tf.keras.activations.softmax)(X)
        
        # Finish Model
        model = tf.keras.models.Model(inputs=inputs, outputs=X)

        return model
        

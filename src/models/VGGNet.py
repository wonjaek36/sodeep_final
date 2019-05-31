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

    def create_keras_model(self, options={}):

        width = self.width
        height = self.height
        channel = self.channel
        output = self.output
        zero_padding = 3

        conv2d_final_width = (width+2*zero_padding) // 2 // 2 // 2 // 2 // 2
        conv2d_final_height = (height+2*zero_padding) // 2 // 2 // 2 // 2 // 2
        flatten_neuron = conv2d_final_height*conv2d_final_width*512
        
        regularization1 = options.get('regularization1',None)
        regularization2 = options.get('regularization2',None)
        regularization3 = options.get('regularization3',None)
        regularization4 = options.get('regularization4',None)
        regularization5 = options.get('regularization5',None)
        regularization6 = options.get('regularization6',None)
        regularization7 = options.get('regularization7',None)
        regularization8 = options.get('regularization8',None)
        
        if regularization1 is not None:
            regularizer1 = tf.keras.regularizers.l2(l=float(regularization1))
        else:
            regularizer1 = None

        if regularization2 is not None:
            regularizer2 = tf.keras.regularizers.l2(l=float(regularization2))
        else:
            regularizer2 = None

        if regularization3 is not None:
            regularizer3 = tf.keras.regularizers.l2(l=float(regularization3))
        else:
            regularizer3 = None

        if regularization4 is not None:
            regularizer4 = tf.keras.regularizers.l2(l=float(regularization4))
        else:
            regularizer4 = None

        if regularization5 is not None:
            regularizer5 = tf.keras.regularizers.l2(l=float(regularization5))
        else:
            regularizer5 = None

        if regularization6 is not None:
            regularizer6 = tf.keras.regularizers.l2(l=float(regularization6))
        else:
            regularizer6 = None

        if regularization7 is not None:
            regularizer7 = tf.keras.regularizers.l2(l=float(regularization7))
        else:
            regularizer7 = None

        if regularization8 is not None:
            regularizer8 = tf.keras.regularizers.l2(l=float(regularization8))
        else:
            regularizer8 = None


        dropout1 = options.get('dropout1',None)
        dropout2 = options.get('dropout2',None)
        dropout3 = options.get('dropout3',None)

        if dropout1 is not None:
            dropout1 = tf.keras.layers.Dropout(float(dropout1))
        else:
            dropout1 = None

        if dropout2 is not None:
            dropout2 = tf.keras.layers.Dropout(float(dropout2))
        else:
            dropout2 = None

        if dropout3 is not None:
            dropout3 = tf.keras.layers.Dropout(float(dropout3))
        else:
            dropout3 = None

        # print (conv2d_final_width)
        # TODO
        # Add kernel initializer "glorot_uniform"

        k_regularizer = tf.keras.regularizers.l2(l=0.01)
        
        # Initialize keras input
        inputs = tf.keras.layers.Input(shape=(width, height, channel))

        # Zero padding 3
        X = tf.keras.layers.ZeroPadding2D(padding=(zero_padding, zero_padding), input_shape=(width, height, channel))(inputs)

        # VGG 1 layer
        X = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='Z1_1', activation='relu', kernel_regularizer=regularizer1, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),  padding='same', activation='relu', name='Z1_2', kernel_regularizer=regularizer1, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

        # VGG 2 layer
        X = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z2_1', kernel_regularizer=regularizer2, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z2_2', kernel_regularizer=regularizer2, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

        # VGG 3 laye
        X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z3_1', kernel_regularizer=regularizer3, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z3_2', kernel_regularizer=regularizer3, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z3_3', kernel_regularizer=regularizer3, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

        # VGG 4 layer
        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z4_1', kernel_regularizer=regularizer4, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z4_2', kernel_regularizer=regularizer4, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z4_3', kernel_regularizer=regularizer4, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(X)

        #VGG 5 layer
        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z5_1', kernel_regularizer=regularizer5, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z5_2', kernel_regularizer=regularizer5, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='Z5_3', kernel_regularizer=regularizer5, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        X = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(X)

        #VGG FC
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(flatten_neuron, activation='relu', name='fc1', kernel_regularizer=regularizer6, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        if dropout1 is not None:
            X = dropout1(X)
        X = tf.keras.layers.Dense(flatten_neuron, activation='relu', name='fc2', kernel_regularizer=regularizer7, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        if dropout2 is not None:
            X = dropout2(X)
        X = tf.keras.layers.Dense(output, activation='softmax', name='fc3', kernel_regularizer=regularizer8, kernel_initializer=tf.initializers.glorot_uniform(), bias_initializer=tf.initializers.zeros())(X)
        if dropout3 is not None:
            X = dropout3(X)

        model = tf.keras.models.Model(inputs=inputs, outputs=X)
        
        return model








        
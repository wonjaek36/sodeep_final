import os
import json
import sys
from parser import Parser
from models.VGGNet import VGGNet
from models.MobileNet import MobileNet
from models.ResNet import ResNet
from models.InceptionV3 import InceptionV3
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
import timeit
import shutil


class Main():
    
    def __init__(self, config):
        self.config = config


    def parse_data(self): 
        
        config = self.config
        data_parser = Parser(config)
        data, labels = data_parser.parsing()
        data_shape, label_shape = data_parser.get_data_description()
        self.data = data
        self.labels = labels
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.num_data = len(labels)

    def get_model(self, model_type=None):
        
        config = self.config

        if model_type is None:
            model_type = config['MODEL']['model']

        if model_type == 'VGGNet':
            model = VGGNet(config)
        elif model_type == 'MobileNet':
            model = MobileNet(config)
        elif model_type == 'ResNet':
            model = ResNet(config)
        elif model_type == 'InceptionV3':
            model = InceptionV3(config)
        else:
            model = None

        return model


    def train(self, options={}):

        start = timeit.default_timer() 
        # model = self.get_model()

        name = options.get("name", None)
        epoch = options.get("epoch", 1000)
        batch_size = options.get("batch_size", 32)
        learning_rate = options.get("learning_rate", 0.0001)
        ratio_train = options.get("train", 0.7)
        ratio_valid = options.get("valid", 0.3)
        model_type = options.get("model", None)
        model = self.get_model(model_type)

        if model is None:
            return

        data_shape = self.data_shape
        output = self.label_shape
        model = model.create_keras_model()
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        numTrain = int(self.num_data * ratio_train)
        numValid = self.num_data - numTrain

        
        # Shuffling data
        idx = np.arange(self.num_data)
        np.random.shuffle(idx)
        data = self.data[idx]
        labels = self.labels[idx]

        x_train = data
        y_train = labels
        #x_train = data[0:numTrain]
        #y_train = labels[0:numTrain]
        #x_valid = data[numTrain:]
        #y_valid = labels[numTrain:]


        if name is not None: 
            # Create directory 
            name = "TEST_" + name

            if os.path.exists(name):
                name = name + "tmp"

            if not os.path.exists(os.path.join('.', name)):
                os.makedirs(os.path.join('.', name))

            with open(os.path.join(name, 'model_summary.txt'), 'w') as f:
                with redirect_stdout(f):
                    model.summary()


            checkpoint_path = name + "/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 period=100,
                                                 verbose=1)

            history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epoch,
                validation_split=0.3,
                shuffle=True,
                callbacks = [cp_callback] )
            
            stop = timeit.default_timer() 
            with open(os.path.join(name, 'model_summary.txt'), 'a') as f:
                f.write('\n')
                f.write('Time: ')
                f.write(str(stop - start))


            history = history.history # Getting history attributes from history instance
            history_filename = "history_" + name + ".txt"
            history_file = open(os.path.join(name, history_filename), 'w')

            history_file.write(str(history['acc']))
            history_file.write('\n')
            history_file.write(str(history['loss']))
            history_file.write('\n')
            history_file.write(str(history['val_acc']))
            history_file.write('\n')
            history_file.write(str(history['val_loss']))
            history_file.write('\n')

            history_file.close()
            
        if name is None:
            model.summary()

            history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epoch,
                validation_data=(x_valid, y_valid),
                shuffle=True)

            stop = timeit.default_timer() 
            print ('Time: ' + str(stop-start))
            history = history.history # Getting history attributes from history instance

            print (history['acc'])
            print (history['loss'])
            print (history['val_acc'])
            print (history['val_loss'])

    def prediction():
        pass

    def load_model():
        pass
        # model.load_weights(checkpoint_path)

if __name__ == "__main__":
    config = None
    
    with open(os.path.join('src', 'config.json'), 'r') as f:
        config = json.load(f)

    if config is not None:
        np.random.seed(1)
        main = Main(config)

        main.parse_data()
        main.train() 
        # main.test()

import os
import json
import sys
from parser import Parser
from models.VGGNet import VGGNet

import numpy as np
import tensorflow as tf

class Main():
    
    def __init__(self, config):
        self.config = config


    def parse_data(self): 
        
        data_parser = Parser(config)
        data, labels = data_parser.parsing()
        data_shape, label_shape = data_parser.get_data_description()
        self.data = data
        self.labels = labels
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.num_data = len(labels)

    def get_model(self):
        
        config = self.config
        model_type = config['MODEL']['model']

        if model_type == 'VGGNet':
            model = VGGNet(config)
        else:
            model = None

        return model

    def train(self):
            
        model = self.get_model()
        if model is None:
            return

        data_shape = self.data_shape
        output = self.label_shape
        #tf.reset_default_graph()
        #with tf.Session() as sess:
            
            # X, Y = model.create_placeholders(data_shape[0], data_shape[1], data_shape[2], output)
            # parameters = model.initialize_parameters()
        model = model.create_keras_model()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        model.summary()


        x_train = self.data[0: int(self.num_data*0.7)]
        y_train = self.labels[0: int(self.num_data*0.7)]
        model.fit(x_train, y_train,
            batch_size=32,
            epochs=1,
            #validation_data=(x_test, y_test),
            shuffle=True)



if __name__ == "__main__":
    config = None
    print (sys.path)
    with open(os.path.join('src', 'config.json'), 'r') as f:
        config = json.load(f)

    if config is not None:
        np.random.seed(1)
        main = Main(config)

        main.parse_data()
        main.train() 
        # main.test()
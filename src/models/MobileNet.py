import tensorflow as tf
from tensorflow import keras
import numpy as np

class MobileNet():

	def __init__(self):
		self.config = config
        self.data_type = config['MODEL']['data']
        data_type = self.data_type
        self.width = config['DATA'][data_type]['width']
        self.height = config['DATA'][data_type]['height']
        self.channel = config['DATA'][data_type]['channel']
        self.output = config['DATA'][data_type]['output']

    def create_keras_model(self):
    	config = self.config
    	
    	data_type = config['MODEL']['data']
        width = self.width
        height = self.height
        channel = self.channel
        output = self.output


"""
MOBILENETV1_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]

"""
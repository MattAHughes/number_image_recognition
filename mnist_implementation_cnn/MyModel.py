# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:15:40 2021

@author: Vitamin-C
"""

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential

class MyModel(tf.keras.Model):
    """A class for CNN's made for MNIST dataset number identification."""
    def __init__(self, use_dp = False, num_output = 10):
        super(MyModel, self).__init__()
        self.use_dp = use_dp
        self.conv_1 = Conv2D(28, (5, 5), activation = tf.nn.relu, padding = 'same')
        self.pooling_1 = MaxPooling2D((2, 2))
        self.conv_2 = Conv2D(28, (5, 5), activation = tf.nn.relu, padding = 'same')
        self.pooling_2 = MaxPooling2D((2, 2))
        self.conv_3 = Conv2D(28, (5, 5), activation = tf.nn.relu, padding = 'same')
        self.pooling_3 = MaxPooling2D((2, 2))
        self.flatten_1 = Flatten()
        self.dense_1 = Dense(250, activation = tf.nn.relu)
        self.categ = Dense(10, activation = tf.nn.softmax)
        if self.use_dp:
            self.use_drop = Dropout(0.2)

    def model(self):
       if self.use_dp:
           model = Sequential([self.conv_1, self.pooling_1,  self.use_drop,
                               self.conv_2, self.pooling_2, self.use_drop,
                               self.conv_3, self.pooling_3, self.use_drop,
                               self.flatten_1, self.dense_1, self.categ,
                               ])
       else:
            model = Sequential([self.conv_1, self.pooling_1, self.conv_2, 
                                self.pooling_2, self.conv_3, self.pooling_3, 
                                self.flatten_1, self.dense_1, self.categ,
                                ])
       return model
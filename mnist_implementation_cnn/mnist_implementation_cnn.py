# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:06:11 2021

@author: Vitamin-C
"""
# import os
# os.chdir('D:\\Projects\\python_work\\tensorflow\\testing_creation_nn')

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg
from sklearn.model_selection import train_test_split
import numpy as np
import json
from contextlib import redirect_stdout
# Import custom classes and modules
from MyModel import MyModel
from results import results
from predict_image_number import predict_number


"""
An attempted implementaion of Savita Ahlawat et al. Improved Handwritten Digit Recognition Using
Convolutional Neural Networks (CNN). Sensors (2020), 20, pg. 3344 to the MNIST dataset.

Set the appropriate dir using greyed import commands.
"""

# Let us use the MNIST Database of numerical images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape for datagen (requires rank 4)
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

# Split the training values into a training and an evaluation set
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train, 
                                              test_size = 0.05,
                                                  random_state = 0)

# Pre-process the data
datagen = idg(rotation_range=15,
               width_shift_range=0.1,
               height_shift_range=0.1,
               horizontal_flip=True,
               featurewise_center=True,
               featurewise_std_normalization = True,
               zca_whitening = True,
               )   

# Define an empty scaler for the data
# Define data normalisation here, since the def is short we won't create a new module
def normalize(x_train, x_val, x_test):
    x_train = x_train.astype("float32")
    x_val = x_val.astype("float32")
    x_test = x_test.astype("float32")
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0
    return x_train, x_val, x_test

x_train, x_val, x_test = normalize(x_train, x_val, x_test)
datagen.fit(x_train)
datagen.fit(x_val)


# One-hot encode the results
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)

# Generate the model with or without (default) dropout
model = MyModel(use_dp = True)
model = model.model()

options = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
                                   epsilon = 1e-7)

model.compile(optimizer=options, loss = tf.losses.categorical_crossentropy,
              metrics = ['accuracy'])

results(model, x_train, y_train,
            x_val, y_val,
            x_test, y_test)

# Save model features
tf.keras.utils.plot_model(model, to_file='mnist_model.png', show_shapes=True)
model.save('mnist_model.h5')

# Save individual config
config = model.get_config()
weights = model.get_weights()
with open('modelsummary.json', 'w') as f:
    with redirect_stdout(f):
        model.summary()

with open("config.json", "w") as file:
   json.dump(config, file)

with open('weights.txt', 'w') as we:
    weights = np.asarray(weights)
    np.savetxt(we, weights, fmt = '%s')

   
# Make some predictions   
predict_number('noisy_3.jpg', model)
predict_number('clean_7.png', model)



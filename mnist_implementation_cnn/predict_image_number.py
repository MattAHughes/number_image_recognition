# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:41:39 2021

@author: Vitamin-C
"""

# Predict the class in a given image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt 
from PIL import Image

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def predict_number(filename, model):
    # use keras preprocessing to resize the image to 28 x 28
    img = load_img(filename, target_size=(28, 28))
    
    # Convert image to greyscale if it has rgb channel
    #img = img.convert(mode="1")
    
    # Show the image to the user
    plt.imshow(img)
    
    # convert image to array
    # reshape into a single sample with 3 channels
    # Use keras im_to_array to get a shapeable array
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)
    
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    # Convert to grayscale for system input (as it expect single channel data)
    img = tf.image.rgb_to_grayscale(img)
    
    #predicting the results
    result = model.predict(img)
    
    # Say what the result means
    dict_1 = {}
    # We have 10 possible values
    for i in range(10):
        dict_1[result[0][i]] = classes[i]
    res = result[0]
    res.sort()
    res = res[::-1]
    results = res[:3]
    print("This image is likely a: ")
    for i in range(3):
        print("{} : {}".format(dict_1[results[i]],
                               (results[i]*100).round(2)))
    print('The image given as input is: ' + filename)
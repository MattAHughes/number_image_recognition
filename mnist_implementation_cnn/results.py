# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:50:47 2021

@author: Vitamin-C
"""
"""
A model for results outputs
"""
import tensorflow as tf
import matplotlib.pyplot as plt


def results(model, x_train, y_train,
            x_val, y_val,
            x_test, y_test):
    # Through initial overtraining we see ~6 epochs is best, though accuracy may increase again rather high epochs
    epoch = 6
    r = model.fit(x_train, y_train, batch_size = 28,
                  epochs = epoch, validation_data =
                  (x_val, y_val), verbose = 1)
    
    acc = model.evaluate(x_test, y_test)
    print("test set loss : ", acc[0])
    print("test set accuracy :", acc[1] * 100)
    epoch_range = range(1, epoch + 1)
    plt.plot(epoch_range, r.history['accuracy'])
    plt.plot(epoch_range, r.history['val_accuracy'])
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()
    # Plot training & validation loss values
    plt.plot(epoch_range,r.history['loss'])
    plt.plot(epoch_range, r.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()
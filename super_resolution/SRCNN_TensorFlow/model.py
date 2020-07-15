import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D,Activation
from tensorflow.keras.optimizers import Adam

def SRCNN(image_size = 33,learning_rate = 1e-4): 
    model = Sequential()

    model.add(Conv2D(64,9,padding='same',input_shape=(image_size,image_size,1)))
    model.add(Activation('relu'))

    model.add(Conv2D(32,1,padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(1,5,padding='same'))
    optimizer = Adam(lr= learning_rate)

    return model
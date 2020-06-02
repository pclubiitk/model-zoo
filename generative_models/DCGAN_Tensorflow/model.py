# Importing Libraries

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense ,Flatten
from tensorflow.keras.layers import  BatchNormalization
from tensorflow.keras.layers import LeakyReLU,ReLU
from tensorflow.keras.layers import Conv2DTranspose,Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
import numpy as np

# Model of Generator

def make_generator(noise_dim):
  model=Sequential()

  model.add(Dense(7*7*256,input_shape=(noise_dim)))
  model.add(BatchNormalization(momentum=0.5))
  model.add(ReLU())
  
  model.add(Reshape((7,7,256)))

  model.add(Conv2DTranspose(128,(3,3),strides=(1,1),padding='same',use_bias='False'))
  model.add(BatchNormalization(momentum=0.5))
  model.add(ReLU())

  model.add(Conv2DTranspose(64,(3,3),strides=(2,2),padding='same',use_bias='False'))
  model.add(BatchNormalization(momentum=0.5))
  model.add(ReLU())

  model.add(Conv2DTranspose(1,(3,3),strides=(2,2),padding='same',use_bias='False',activation='tanh'))
  return model

# Model of discriminator

def make_discriminator():
  model=Sequential()

  model.add(Conv2D(64,(3,3),strides=(2,2),padding='same',input_shape=[28,28,1]))
  model.add(LeakyReLU(alpha= 0.2))
  model.add(Dropout(0.3))

  model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
  model.add(BatchNormalization(momentum=0.5))
  model.add(LeakyReLU(alpha= 0.2))
  model.add(Dropout(0.3))

  model.add(Conv2D(256,(3,3),strides=(2,2),padding='same'))
  model.add(BatchNormalization(momentum=0.5))
  model.add(LeakyReLU(alpha= 0.2))
  model.add(Dropout(0.3))

  model.add(Conv2D(512,(3,3),strides=(2,2),padding='same'))
  model.add(BatchNormalization(momentum=0.5))
  model.add(LeakyReLU(alpha= 0.2))
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(1))

  return model



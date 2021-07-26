import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    SeparableConv2D,
    Conv2D,
    Activation,
    BatchNormalization,
    GlobalAveragePooling2D,Add, MaxPooling2D
)
from tensorflow.keras.optimizers import Adam

def conv_bn(X, filters, kernel_size, strides=1):
  X = Conv2D(filters=filters, kernel_size=kernel_size, strides = strides, use_bias=False, padding='same')(X)
  X = BatchNormalization()(X)
  return X

def Sepconv_bn(X, filters, kernel_size, strides):
  X = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides = strides, padding='same', use_bias=False)(X)
  X = BatchNormalization()(X)  
  return X  

def entry_flow(X):

  X  = conv_bn(X, filters=32, kernel_size=3, strides=2)
  X = Activation('relu')(X)
  X = conv_bn(X, filters=64, kernel_size=3, strides=1)
  point = Activation('relu')(X)

  X = Sepconv_bn(point, filters=128, kernel_size=3, strides=1)
  X = Activation('relu')(X)
  X = Sepconv_bn(X, filters=128, kernel_size=3, strides=1)
  X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
  point = conv_bn(point, filters=128, strides=2, kernel_size=1)

  X = Add()([X,point])

  X = Activation('relu')(X)
  X = Sepconv_bn(X, filters=256, kernel_size=3, strides=1)
  X = Activation('relu')(X)
  X = Sepconv_bn(X, filters=256, kernel_size=3, strides=1)
  X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

  point = conv_bn(point, filters=256, strides=2, kernel_size=1)
  X = Add()([X,point])

  X = Activation('relu')(X)
  X = Sepconv_bn(X, filters=728, kernel_size=3, strides=1)
  X = Activation('relu')(X)
  X = Sepconv_bn(X, filters=728, kernel_size=3, strides=1)
  X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

  point = conv_bn(point, filters=728, strides=2, kernel_size=1)
  X = Add()([X,point])

  return X

def middle_flow(X):
  for i in range(8):
      T = Activation('relu')(X)
      T = Sepconv_bn(T, filters=728, kernel_size=3, strides=1)
      T = Activation('relu')(T)
      T = Sepconv_bn(T, filters=728, kernel_size=3, strides=1)
      T = Activation('relu')(T)
      T = Sepconv_bn(T, filters=728, kernel_size=3, strides=1)
      X = Add()([X,T])
  return X

def exit_flow(X):
  
  point = X
  X = Activation('relu')(X)
  X = Sepconv_bn(X, filters=728, kernel_size=3, strides=1)
  X = Activation('relu')(X)
  X = Sepconv_bn(X, filters=1024, kernel_size=3, strides=1)
  X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

  point = conv_bn(point, filters=1024, strides=2, kernel_size=1)
  X = Add()([X,point])

  X = Sepconv_bn(X, filters=1536, kernel_size=3, strides=1)
  X = Activation('relu')(X)
  X = Sepconv_bn(X, filters=2048, kernel_size=3, strides=1)
  X = Activation('relu')(X)

  X = GlobalAveragePooling2D()(X)
  X = Dense(units=10, activation='softmax')(X)
  return X


def Xception(input):
   X = entry_flow(input)
   X = middle_flow(X)
   outputs = exit_flow(X)
   model = Model(input,outputs)
   return model


#plot_model(model,show_shapes=True)
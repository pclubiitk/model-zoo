import numpy as np

import keras
from keras.layers import Reshape,LeakyReLU ,  Conv2D , Dense ,Input , Lambda , Conv2DTranspose , Flatten , Dropout , Activation 
from keras.models import Model
from keras.datasets.mnist import load_data
from keras import backend
from keras.optimizers import Adam
from matplotlib import pyplot

def define_generator(latent_dim,alpha_):
  input_ = Input(shape=(latent_dim,))
  nodes = 7*7*128
  generator = Dense(nodes)(input_)
  generator = LeakyReLU(alpha = alpha_)(generator)
  generator = Reshape((7,7,128))(generator)
  
  generator = Conv2DTranspose(128 ,kernel_size= (4,4), strides = (2,2) , padding = 'same' )(generator)
  generator = LeakyReLU(alpha = alpha_)(generator)

  generator = Conv2DTranspose(128 ,kernel_size= (4,4), strides = (2,2) , padding = 'same' )(generator)
  generator = LeakyReLU(alpha = alpha_)(generator)

  final_layer = Conv2D(1 ,(7,7),activation = 'tanh',padding ='same' )(generator)
  print(input_.shape)
  model = Model(input_ , final_layer)
  return model


def custom_activation(output):
  logexp_sum = backend.sum(backend.exp(output), axis =-1 , keepdims =True)
  result = logexp_sum / (logexp_sum+1.0)
  return result

def define_discriminator(alpha_,dropout_,lr_,beta_1_,input_shape=(28,28,1), num_classes =10):
  input_img = Input(shape = input_shape)
  dis = Conv2D(128 , (3,3) , strides =(2,2) , padding = 'same')(input_img)
  dis = LeakyReLU(alpha = alpha_)(dis)

  dis = Conv2D(128 , (3,3) , strides =(2,2) , padding = 'same')(dis)
  dis = LeakyReLU(alpha = alpha_)(dis)

  dis = Conv2D(128 , (3,3) , strides =(2,2) , padding = 'same')(dis)
  dis = LeakyReLU(alpha = alpha_)(dis)

  dis = Flatten()(dis)
  dis = Dropout(dropout_)(dis)
  dis = Dense(num_classes)(dis)
  #### supervised output
  s_out_layer = Activation('softmax')(dis)
  s_model = Model(input_img , s_out_layer)
  s_model.compile(loss='sparse_categorical_crossentropy',optimizer = Adam(lr=lr_, beta_1=beta_1_) , metrics =['accuracy'] )
  #### unsupervised output
  us_out_layer = Lambda(custom_activation)(dis)
  us_model = Model(input_img , us_out_layer)
  us_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr_, beta_1=beta_1_))
  return s_model , us_model

def define_gan(g_model , dis_model,lr_,beta_1_):
  dis_model.trainable = False
  gan_output = dis_model(g_model.output)
  gan_model = Model(g_model.input , gan_output)
  gan_model.compile(loss = 'binary_crossentropy' , optimizer = Adam(lr = lr_ , beta_1 =beta_1_))
  return gan_model 
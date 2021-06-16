"""
Stage-1 model of StackGAN
"""

import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Reshape, UpSampling2D, Conv2D, Activation, \
    concatenate, Flatten, Lambda, Concatenate
from keras import backend as K


class Stage1_Model:

  def __init__(self) -> None:
      pass

  def generate_c(self,x):
    """
    Obtain Text Conditioning Variable 
    """
    mean = x[:, :128]
    log_sigma = x[:, 128:]
    stddev = K.exp(log_sigma)
    epsilon = K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
    c = stddev * epsilon + mean
    return c

  def build_ca_model(self):
    """
    Get conditioning augmentation model.
    Takes an embedding of shape (1024,) and returns a tensor of shape (256,)
    """
    input_layer = Input(shape=(1024,))
    x = Dense(256)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    model = Model(inputs=[input_layer], outputs=[x])
    return model

  def build_embedding_compressor_model(self):
    """
    Builds embedding compressor model
    """
    input_layer = Input(shape=(1024,))
    x = Dense(128)(input_layer)
    x = ReLU()(x)

    model = Model(inputs=[input_layer], outputs=[x])
    return model

  def build_stage1_generator(self):
      """
      Builds a generator model used in Stage-I
      """
      input_layer = Input(shape=(1024,))
      x = Dense(256)(input_layer)
      mean_logsigma = LeakyReLU(alpha=0.2)(x)
  
      c = Lambda(self.generate_c)(mean_logsigma)
  
      input_layer2 = Input(shape=(100,))
  
      gen_input = Concatenate(axis=1)([c, input_layer2])
  
      x = Dense(128 * 8 * 4 * 4, use_bias=False)(gen_input)
      x = ReLU()(x)
  
      x = Reshape((4, 4, 128 * 8), input_shape=(128 * 8 * 4 * 4,))(x)
  
      x = UpSampling2D(size=(2, 2))(x)
      x = Conv2D(512, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)
  
      x = UpSampling2D(size=(2, 2))(x)
      x = Conv2D(256, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)
  
      x = UpSampling2D(size=(2, 2))(x)
      x = Conv2D(128, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)
  
      x = UpSampling2D(size=(2, 2))(x)
      x = Conv2D(64, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)
  
      x = Conv2D(3, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
      x = Activation(activation='tanh')(x)
  
      stage1_gen = Model(inputs=[input_layer, input_layer2], outputs=[x, mean_logsigma])
      return stage1_gen
  
  def build_stage1_discriminator(self):
      """
      Create a model which takes two inputs
      1. One from the generator network of stage1
      2. One from the embedding layer
      3. Concatenate along the axis dimension and feed it to the last module which produces final logits
      """
      input_layer = Input(shape=(64, 64, 3))
  
      x = Conv2D(64, (4, 4),
                 padding='same', strides=2,
                 input_shape=(64, 64, 3), use_bias=False)(input_layer)
      x = LeakyReLU(alpha=0.2)(x)
  
      x = Conv2D(128, (4, 4), padding='same', strides=2, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(alpha=0.2)(x)
  
      x = Conv2D(256, (4, 4), padding='same', strides=2, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(alpha=0.2)(x)
  
      x = Conv2D(512, (4, 4), padding='same', strides=2, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(alpha=0.2)(x)
  
      input_layer2 = Input(shape=(4, 4, 128))
  
      merged_input = concatenate([x, input_layer2])
  
      x2 = Conv2D(64 * 8, kernel_size=1,
                  padding="same", strides=1)(merged_input)
      x2 = BatchNormalization()(x2)
      x2 = LeakyReLU(alpha=0.2)(x2)
      x2 = Flatten()(x2)
      x2 = Dense(1)(x2)
      x2 = Activation('sigmoid')(x2)
  
      stage1_dis = Model(inputs=[input_layer, input_layer2], outputs=[x2])
      return stage1_dis
  
  def build_adversarial_model(self, gen_model, dis_model):
      input_layer = Input(shape=(1024,))
      input_layer2 = Input(shape=(100,))
      input_layer3 = Input(shape=(4, 4, 128))
  
      x, mean_logsigma = gen_model([input_layer, input_layer2])
  
      dis_model.trainable = False
      valid = dis_model([x, input_layer3])
  
      model = Model(inputs=[input_layer, input_layer2, input_layer3], outputs=[valid, mean_logsigma])
      return model
"""
Stage-2 model of StackGAN
"""

import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Reshape, UpSampling2D, Conv2D, Activation, \
    concatenate, Flatten, Lambda, Concatenate, ZeroPadding2D, add
from keras import backend as K


class Stage2_Model:

  def __init__(self) -> None:
      pass

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
    Build embedding compressor model
    """
    input_layer = Input(shape=(1024,))
    x = Dense(128)(input_layer)
    x = ReLU()(x)
    model = Model(inputs=[input_layer], outputs=[x])
    return model

  def generate_c(self, x):
    mean = x[:, :128]
    log_sigma = x[:, 128:]

    stddev = K.exp(log_sigma)
    epsilon = K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
    c = stddev * epsilon + mean
    return c

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

  def residual_block(self,input):
      """
      Residual block in the generator network
      """
      x = Conv2D(128 * 4, kernel_size=(3, 3), padding='same', strides=1)(input)
      x = BatchNormalization()(x)
      x = ReLU()(x)

      x = Conv2D(128 * 4, kernel_size=(3, 3), strides=1, padding='same')(x)
      x = BatchNormalization()(x)

      x = add([x, input])
      x = ReLU()(x)

      return x

  def joint_block(self,inputs):
      c = inputs[0]
      x = inputs[1]

      c = K.expand_dims(c, axis=1)
      c = K.expand_dims(c, axis=1)
      c = K.tile(c, [1, 16, 16, 1])
      return K.concatenate([c, x], axis=3)

  def build_stage2_generator(self):
      """
      Create Stage-II generator containing the CA Augmentation Network,
      the image encoder and the generator network
      """

      # 1. CA Augmentation Network
      input_layer = Input(shape=(1024,))
      input_lr_images = Input(shape=(64, 64, 3))

      ca = Dense(256)(input_layer)
      mean_logsigma = LeakyReLU(alpha=0.2)(ca)
      c = Lambda(self.generate_c)(mean_logsigma)

      # 2. Image Encoder
      x = ZeroPadding2D(padding=(1, 1))(input_lr_images)
      x = Conv2D(128, kernel_size=(3, 3), strides=1, use_bias=False)(x)
      x = ReLU()(x)

      x = ZeroPadding2D(padding=(1, 1))(x)
      x = Conv2D(256, kernel_size=(4, 4), strides=2, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)

      x = ZeroPadding2D(padding=(1, 1))(x)
      x = Conv2D(512, kernel_size=(4, 4), strides=2, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)

      # 3. Joint
      c_code = Lambda(self.joint_block)([c, x])

      x = ZeroPadding2D(padding=(1, 1))(c_code)
      x = Conv2D(512, kernel_size=(3, 3), strides=1, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)

      # 4. Residual blocks
      x = self.residual_block(x)
      x = self.residual_block(x)
      x = self.residual_block(x)
      x = self.residual_block(x)

      # 5. Upsampling blocks
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
      x = Activation('tanh')(x)

      model = Model(inputs=[input_layer, input_lr_images], outputs=[x, mean_logsigma])
      return model

  def build_stage2_discriminator(self):
      """
      Create Stage-II discriminator network
      """
      input_layer = Input(shape=(256, 256, 3))

      x = Conv2D(64, (4, 4), padding='same', strides=2, input_shape=(256, 256, 3), use_bias=False)(input_layer)
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

      x = Conv2D(1024, (4, 4), padding='same', strides=2, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(alpha=0.2)(x)

      x = Conv2D(2048, (4, 4), padding='same', strides=2, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(alpha=0.2)(x)

      x = Conv2D(1024, (1, 1), padding='same', strides=1, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(alpha=0.2)(x)

      x = Conv2D(512, (1, 1), padding='same', strides=1, use_bias=False)(x)
      x = BatchNormalization()(x)

      x2 = Conv2D(128, (1, 1), padding='same', strides=1, use_bias=False)(x)
      x2 = BatchNormalization()(x2)
      x2 = LeakyReLU(alpha=0.2)(x2)

      x2 = Conv2D(128, (3, 3), padding='same', strides=1, use_bias=False)(x2)
      x2 = BatchNormalization()(x2)
      x2 = LeakyReLU(alpha=0.2)(x2)

      x2 = Conv2D(512, (3, 3), padding='same', strides=1, use_bias=False)(x2)
      x2 = BatchNormalization()(x2)

      added_x = add([x, x2])
      added_x = LeakyReLU(alpha=0.2)(added_x)

      input_layer2 = Input(shape=(4, 4, 128))

      merged_input = concatenate([added_x, input_layer2])

      x3 = Conv2D(64 * 8, kernel_size=1, padding="same", strides=1)(merged_input)
      x3 = BatchNormalization()(x3)
      x3 = LeakyReLU(alpha=0.2)(x3)
      x3 = Flatten()(x3)
      x3 = Dense(1)(x3)
      x3 = Activation('sigmoid')(x3)

      stage2_dis = Model(inputs=[input_layer, input_layer2], outputs=[x3])
      return stage2_dis

  def build_adversarial_model(self, gen_model2, dis_model, gen_model1):
      """
      Create adversarial model
      """
      embeddings_input_layer = Input(shape=(1024, ))
      noise_input_layer = Input(shape=(100, ))
      compressed_embedding_input_layer = Input(shape=(4, 4, 128))

      gen_model1.trainable = False
      dis_model.trainable = False

      lr_images, mean_logsigma1 = gen_model1([embeddings_input_layer, noise_input_layer])
      hr_images, mean_logsigma2 = gen_model2([embeddings_input_layer, lr_images])
      valid = dis_model([hr_images, compressed_embedding_input_layer])

      model = Model(inputs=[embeddings_input_layer, noise_input_layer, compressed_embedding_input_layer], outputs=[valid, mean_logsigma2])
      return model
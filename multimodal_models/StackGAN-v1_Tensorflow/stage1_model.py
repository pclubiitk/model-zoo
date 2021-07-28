"""
Stage-1 model of StackGAN
"""

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import layers


class Stage1_Model:

  def __init__(self) -> None:
      pass

  def generate_c(self,x):
    """
    Obtain Text Conditioning Variable 
    """
    mean = x[:, :128]
    log_sigma = x[:, 128:]
    stddev = tf.math.exp(log_sigma)
    epsilon = tf.random.normal(shape=(mean.shape[1],), dtype=tf.dtypes.float32)
    c = mean + epsilon*stddev
    return c


  def build_stage1_generator(self):
      """
      Stage-I Generator Model

      Input:
        Embedded Text Description of shape (1024,)
        Random Noise of shape (100,)

      Output:
        Generated Image of shape (64,64,3) and concatenated mean and logsigma of shape (256,)
      """
      input_layer1 = Input(shape=(1024,))
      x = layers.Dense(256)(input_layer1)
      mean_logsigma = layers.LeakyReLU(alpha=0.2)(x)
  
      c = layers.Lambda(self.generate_c)(mean_logsigma)
  
      input_layer2 = Input(shape=(100,))      # Random noise vector of shape (100,)
  
      concat_input = layers.Concatenate(axis=1)([c, input_layer2])
  
      x = layers.Dense(128 * 8 * 4 * 4, kernel_initializer="glorot_normal")(concat_input)
      x = layers.ReLU()(x)
  
      x = layers.Reshape((4, 4, 128 * 8), input_shape=(128 * 8 * 4 * 4,))(x)
  
      x = layers.UpSampling2D(size=(2, 2))(x)
      x = layers.Conv2D(512, kernel_size=3, padding="same", strides=1, kernel_initializer="glorot_normal")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
  
      x = layers.UpSampling2D(size=(2, 2))(x)
      x = layers.Conv2D(256, kernel_size=3, padding="same", strides=1, kernel_initializer="glorot_normal")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
  
      x = layers.UpSampling2D(size=(2, 2))(x)
      x = layers.Conv2D(128, kernel_size=3, padding="same", strides=1, kernel_initializer="glorot_normal")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
  
      x = layers.UpSampling2D(size=(2, 2))(x)
      x = layers.Conv2D(64, kernel_size=3, padding="same", strides=1, kernel_initializer="glorot_normal")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
  
      x = layers.Conv2D(3, kernel_size=3, padding="same", strides=1, kernel_initializer="glorot_normal")(x)
      x = layers.Activation(activation='tanh')(x)
  
      stage1_gen = Model(inputs=[input_layer1, input_layer2], outputs=[x, mean_logsigma])
      return stage1_gen
  

  def build_stage1_discriminator(self):
      """
      Stage-I Discriminator Model
      Input:
        Text Embedding of shape (1024,)
        Image of shape (64,64,3), either fake or real

      Output:
        Classification whether the image is real (1) or fake (0)
      """
      input_layer1 = Input(shape=(64, 64, 3))

      x = layers.ZeroPadding2D(padding=(1, 1))(input_layer1)
      x = layers.Conv2D(64, (4, 4),
                 padding='valid', strides=2,
                 input_shape=(64, 64, 3), kernel_initializer="glorot_normal")(x)
      x = layers.LeakyReLU(alpha=0.2)(x)

      x = layers.ZeroPadding2D(padding=(1, 1))(x)
      x = layers.Conv2D(128, (4, 4), padding='valid', strides=2, kernel_initializer="glorot_normal")(x)
      x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU(alpha=0.2)(x)
  
      x = layers.ZeroPadding2D(padding=(1, 1))(x)
      x = layers.Conv2D(256, (4, 4), padding='valid', strides=2, kernel_initializer="glorot_normal")(x)
      x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU(alpha=0.2)(x)
  
      x = layers.ZeroPadding2D(padding=(1, 1))(x)
      x = layers.Conv2D(512, (4, 4), padding='valid', strides=2, kernel_initializer="glorot_normal")(x)
      x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU(alpha=0.2)(x)
  
      input_layer2 = Input(shape=(1024,))
      compressed_embedding = layers.Dense(128)(input_layer2)
      compressed_embedding = layers.ReLU()(compressed_embedding)
      compressed_embedding = tf.reshape(compressed_embedding, (-1, 1, 1, 128))
      compressed_embedding = tf.tile(compressed_embedding, (1, 4, 4, 1))
  
      merged_input = layers.concatenate([x, compressed_embedding])
  
      x2 = layers.Conv2D(64 * 8, kernel_size=1,
                  padding="same", strides=1, kernel_initializer="glorot_normal")(merged_input)
      x2 = layers.BatchNormalization()(x2)
      x2 = layers.LeakyReLU(alpha=0.2)(x2)
      x2 = layers.Flatten()(x2)
      x2 = layers.Dense(1)(x2)
      x2 = layers.Activation('sigmoid')(x2)
  
      stage1_dis = Model(inputs=[input_layer1, input_layer2], outputs=[x2])
      return stage1_dis

  
  def build_adversarial_model(self, gen_model, dis_model):
      input_layer1 = Input(shape=(1024,))
      input_layer2 = Input(shape=(100,))
      input_layer3 = Input(shape=(1024,))
  
      x, mean_logsigma = gen_model([input_layer1, input_layer2])
  
      dis_model.trainable = False
      logit = dis_model([x, input_layer3])
  
      model = Model(inputs=[input_layer1, input_layer2, input_layer3], outputs=[logit, mean_logsigma])
      return model
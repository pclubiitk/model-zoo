import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model(noise_dim = 64, categorical_dim = 10, continuous_dim = 2):
    model = tf.keras.Sequential()
    
    #Layer 1: 4*4*256, Reshape to 4x4x256
    model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(noise_dim + categorical_dim + continuous_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))
    model.add(layers.Reshape((4, 4, 256)))
    
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None,32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))

    model.add(layers.Conv2D(3, (4, 4), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3)

    return model
    
def make_discriminator_model(categorical_dim = 10, continous_dim = 2):
    model = tf.keras.Sequential()

    #Layer 1: Conv2D 4x4, output : 16x16x64
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[32, 32, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))
  
    #Layer 2: Conv2D 4x4, output : 8x8x128
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))

    #Layer 3: Conv2D 4x4, output : 4x4x256  
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))

    #Layer 4: Dense 128
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    
    #Layer 5: prediction layer: Dense (1 + categorical_dim + continous_dim)
    model.add(layers.Dense(1 + categorical_dim + continous_dim))

    return model
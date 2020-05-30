import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model(noise_dim = 62, categorical_dim = 10, continuous_dim = 2):
    model = tf.keras.Sequential()

    # Layer 1: Dense 1024
    model.add(layers.Dense(1024, use_bias=False, input_shape=(noise_dim + categorical_dim + continuous_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Layer 2: Dense 7*7*128
    model.add(layers.Dense(7*7*128))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Reshape output to 7x7x128 image
    model.add(layers.Reshape((7, 7, 128)))
    assert model.output_shape == (None, 7, 7, 128) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
    
def make_discriminator_model(categorical_dim = 10, continous_dim = 2):
    model = tf.keras.Sequential()

    #Layer 1: Conv2D 4x4, output : 14x14x64
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))
  
    #Layer 2: Conv2D 4x4, output : 7x7x128
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))

    #Layer 3: Dense 128
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.1))

    #Layer 4: prediction layer: Dense (1 + categorical_dim + continous_dim)
    model.add(layers.Dense(1 + categorical_dim + continous_dim))

    return model
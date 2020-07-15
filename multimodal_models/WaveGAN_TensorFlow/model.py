import tensorflow as tf
from tensorflow.keras import layers
from utils import apply_phaseshuffle

def Generator(kernel_len=(5,5),dim=64,channels=1):
    model = tf.keras.Sequential(name='generator')
    #Input Layer
    model.add(layers.Dense(4 * 4 * dim * 16, use_bias=False, input_shape=(100,)))
    model.add(layers.Reshape((4,4,dim*16)))
    model.add(layers.ReLU())
    #Layer 0
    model.add(layers.Conv2DTranspose(dim * 8, kernel_len, strides=(2,2), padding='same', use_bias=False, activation='relu'))
    model.add(layers.ReLU())
    #Layer 1
    model.add(layers.Conv2DTranspose(dim * 4, kernel_len, strides=(2,2), padding='same', use_bias=False, activation='relu'))
    model.add(layers.ReLU())
    #Layer 2
    model.add(layers.Conv2DTranspose(dim * 2, kernel_len, strides=(2,2), padding='same', use_bias=False, activation='relu'))
    model.add(layers.ReLU())
    #Layer 3
    model.add(layers.Conv2DTranspose(dim, kernel_len, strides=(2,2), padding='same', use_bias=False, activation='relu'))
    model.add(layers.ReLU())
    #Layer 4
    model.add(layers.Conv2DTranspose(channels, kernel_len, strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    model.add(layers.Lambda(lambda x: tf.nn.tanh(x), name='Tanh'))

    return model


def Discrimiator(kernel_len=(5,5),dim=64,channels=1,phaseshuffle_rad=2):
    model = tf.keras.Sequential(name='discriminator')
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
    #Layer 0
    model.add(layers.Conv2D(dim, kernel_len, strides=(2,2), padding='same', use_bias=False, input_shape=(128,128,channels)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Lambda(phaseshuffle, name='phase_shuffler0'))
    #Layer 1
    model.add(layers.Conv2D(dim * 2, kernel_len, strides=(2,2), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Lambda(phaseshuffle, name='phase_shuffler1'))
    #Layer 2
    model.add(layers.Conv2D(dim * 4, kernel_len, strides=(2,2), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Lambda(phaseshuffle, name='phase_shuffler2'))
    #Layer 3
    model.add(layers.Conv2D(dim * 8, kernel_len, strides=(2,2), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Lambda(phaseshuffle, name='phase_shuffler3'))
    #Layer 4
    model.add(layers.Conv2D(dim * 16, kernel_len, strides=(2,2), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.2))
    #Layer 5
    model.add(layers.Reshape((4*4*dim*16,)))
    model.add(layers.Dense(1))

    return model



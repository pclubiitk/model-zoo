import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def make_generator(inputs, labels, image_size):

    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = layers.concatenate([inputs, labels], axis=1)
    x = layers.Dense(image_resize * image_resize * layer_filters[0])(x)
    x = layers.Reshape((image_resize, image_resize, layer_filters[0]))(x)
    

    for filters in layer_filters:

        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same')(x)

    x = layers.Activation('sigmoid')(x)

    generator = keras.models.Model([inputs, labels], x, name='generator')
    return generator

def make_discriminator(inputs, labels, image_size):

    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs

    y = layers.Dense(image_size * image_size)(labels)
    y = layers.Reshape((image_size, image_size, 1))(y)
    x = layers.concatenate([x, y])

    for filters in layer_filters:

        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2

        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    x = layers.Activation('sigmoid')(x)
    
    discriminator = keras.models.Model([inputs, labels], x, name='discriminator')
    return discriminator

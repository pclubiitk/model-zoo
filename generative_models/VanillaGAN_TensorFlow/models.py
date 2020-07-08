import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization,LeakyReLU
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam

def adam_optimizer(learning_rate,beta_1):
    return Adam(lr=learning_rate, beta_1=beta_1)

def create_generator(learning_rate,beta_1,encoding_dims):

    generator=Sequential()

    generator.add(Dense(units=256,input_dim=encoding_dims))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(units=784, activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(learning_rate,beta_1))
    return generator

def create_discriminator(learning_rate,beta_1):

    discriminator=Sequential()

    discriminator.add(Dense(units=1024,input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(units=1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(learning_rate,beta_1))
    return discriminator

def create_gan(discriminator, generator,encoding_dims):

    discriminator.trainable=False
    gan_input = Input(shape=(encoding_dims,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

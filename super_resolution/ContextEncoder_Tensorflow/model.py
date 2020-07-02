from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Activation, BatchNormalization, LeakyReLU, Flatten, Dense, Dropout, UpSampling2D
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf


def define_discriminator(in_shape=(16,16,3)):
	model = Sequential()
	model.add(Conv2D(64, kernel_size=4, strides=2, padding="same",input_shape=in_shape,name="dconv1",kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	model.add(Conv2D(128, kernel_size=4, strides=2, padding="same",name="dconv2",kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=4,strides=2, padding="same",name="dconv3",kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(1,activation="sigmoid"))
	return model

def define_generator(in_shape=(32,32,3),channels=3):
	model = Sequential()
	# Encoder
	model.add(Conv2D(128, kernel_size=4, strides=2, padding="same",name="conv1",input_shape=in_shape,kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=4, strides=2, padding="same",name="conv2",kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Conv2D(512, kernel_size=4, strides=2, padding="same",name="conv3",kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
		
	model.add(Conv2D(4000,kernel_size=4,name="conv5"))

	# Decoder
	model.add(Conv2DTranspose(512,kernel_size=4,name="conv6",strides=2))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(256, kernel_size=4,strides=2,name="conv7",padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	model.add(Conv2DTranspose(channels, kernel_size=4,strides=2,name="conv8",padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
	model.add(Activation('tanh'))

	return model

def define_gan(generator, discriminator):
	discriminator.trainable = False
	model = Sequential()
	model.add(generator)
	model.add(discriminator)

	return model

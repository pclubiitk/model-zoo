import tensorflow as tf
import numpy as np

def dataloader(dataset):

  if dataset=="mnist":
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    num_labels = np.amax(y_train) + 1
    y_train = tf.keras.utils.to_categorical(y_train)
    return x_train, y_train, image_size, num_labels

  if dataset=="cifar10":
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    num_labels = np.amax(y_train) + 1
    y_train = tf.keras.utils.to_categorical(y_train)
    return x_train, y_train, image_size, num_labels

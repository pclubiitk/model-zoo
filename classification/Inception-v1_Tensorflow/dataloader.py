import tensorflow as tf
import numpy as np


def load():
    (trainIm, trainLab), (testIm, testLab) = tf.keras.datasets.cifar10.load_data()
    trainIm = trainIm / 255.0
    testIm = testIm / 255.0

    # Subtract the mean from the Dataset
    trainImMean = np.mean(trainIm, axis=0)
    trainIm -= trainImMean
    testIm -= trainImMean

    trainLab = tf.keras.utils.to_categorical(trainLab, 10)
    testLab = tf.keras.utils.to_categorical(testLab, 10)

    return (trainIm, trainLab, testIm, testLab)

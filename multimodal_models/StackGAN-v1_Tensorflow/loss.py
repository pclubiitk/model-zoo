"""
Module to calculate different losses, log the losses and save the rgb image
"""

import tensorflow as tf
import matplotlib.pyplot as plt


def KL_loss(y_true, y_pred):
    """
    Calculate Kullback–Leibler divergence
    """
    mean = y_pred[:, :128]
    logsigma = y_pred[:, :128]
    loss = -logsigma + .5 * ( tf.math.square(mean) -1 + tf.math.exp(2. * logsigma))
    loss = tf.math.reduce_mean(loss)
    return loss


def custom_generator_loss(y_true, y_pred):
    """
    Calculate binary cross entropy loss
    """
    return tf.keras.metrics.binary_crossentropy(y_true, y_pred)


def save_rgb_img(img, path):
    """
    Save an rgb image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()
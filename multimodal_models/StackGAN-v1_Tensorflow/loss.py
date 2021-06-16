"""
Module to calculate different losses, log the losses and save the rgb image
"""


import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt

def KL_loss(y_true, y_pred):
    """
    Calculate Kullback–Leibler divergence
    """
    mean = y_pred[:, :128]
    logsigma = y_pred[:, :128]
    loss = -logsigma + .5 * (-1 + K.exp(2. * logsigma) + K.square(mean))
    loss = K.mean(loss)
    return loss

def custom_generator_loss(y_true, y_pred):
    """
    Calculate binary cross entropy loss
    """
    return K.binary_crossentropy(y_true, y_pred)

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

def write_log(callback, name, loss, batch_no, logdir):
    """
    Write training summary to TensorBoard
    """
    summary = tf.summary.create_file_writer(logdir)
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

    # summary = tf.summary.create_file_writer("/tmp/mylogs")
    # with writer.as_default():
    #   for step in range(100):
    #   # other model code would go here
    #   tf.summary.scalar("my_metric", 0.5, step=step)
    #   writer.flush()
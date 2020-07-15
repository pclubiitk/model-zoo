import tensorflow as tf
import numpy as np

def get_dataset(args):
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = (x_train - 127.5) / 255.0
    x_train = x_train[..., tf.newaxis].astype(np.float32)
    args.data_shape = [28, 28, 1]

    train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(args.batch_size)
    return train_ds

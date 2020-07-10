import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_data():
    (x_train,_), (_,_) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    x_train = x_train.reshape(60000, 784)
    return x_train

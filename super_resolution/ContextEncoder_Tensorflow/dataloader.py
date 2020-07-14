import tensorflow as tf
import numpy as np

def get_dataset(args):
    
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), _ = cifar10.load_data()
    x_train = (x_train - 127.5) / 255.0
    args.data_shape = x_train.shape[1:]

    idx = np.random.randint(0, len(x_train), args.num_img)
    validation = x_train[idx]
  
    train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(args.batch_size)
    return train_ds,validation

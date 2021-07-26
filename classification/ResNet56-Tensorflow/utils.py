import tensorflow as tf
import numpy as np
import hyperparams as hprms



def preprocess_data(X_train, y_train, X_test, y_test):

  X_train = X_train.astype(np.float32)      # Convert to float32 if not
  X_test = X_test.astype(np.float32)

  X_train = X_train/255.    # Min-Max Scaling
  X_test = X_test/255.

  y_train = tf.one_hot(y_train.reshape((len(y_train,))), depth=10)    # One hot encode target values
  y_test = tf.one_hot(y_test.reshape((len(y_test,))), depth=10)

  return X_train, y_train, X_test, y_test


def generate_random_mini_batches(X, y):

  m = len(X)
  batch_size = hprms.batch_size

  n_batches = m//batch_size
  batches = []

  for i in range(n_batches):
    # curr["X"] = X[i*batch_size:(i+1)*batch_size]
    #  curr["y"] = y[i*batch_size:(i+1)*batch_size]
     batches.append((X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]))

  if m%batch_size == 0:
    return batches

  else :
    i = n_batches
    batches.append((X[i*batch_size:], y[i*batch_size:]))
    return batches

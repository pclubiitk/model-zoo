import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import os

from load_dataset import (
    # Note you have to run 'load_dataset.py' atleast once so that you've downloaded the dataset
    # This import is to load the database that we already downloaded, we're using the CIFAR-100 dataset.
    # Making use     of keras to load the dataset. (check 'load_dataset.py')
    x_train,
    x_test,
    y_test,
    y_train,
)

from utils import (
    # Importing functions and other objects from utils.py
    cifar_dict,
)

from model import (
    # Function to build the model
    train_model,
    build_model,
    train_bool,
    test_display,
    checkpoint_path,
)


# Preparing the model
my_model = build_model()
# Printing the summary of the model
my_model.summary()

# The below line does training if train_bool is set to True [You can change this in model.py]
if train_bool:
    my_model = train_model(my_model, checkpoint_path)

#  Loading any checkpoint weights if we have any
tmp_checkpoint_dir = os.path.dirname(
    "/home/merp/Desktop/model-zoo/classification/MLP-Mixer_tensorflow/models/cp.ckpt"
)
latest_weights = tf.train.latest_checkpoint(tmp_checkpoint_dir)
# If we have a saved weights in train folder
if not ((latest_weights) == None):
    my_model.load_weights(latest_weights)

# To display the model's performance on a random image.
if test_display:
    import random

    i = random.randint(1, 10000)
    plt.imshow(x_test[i])
    copy = x_test[i]
    copy = copy[None, :, :, :]
    y_pred = my_model.predict(copy)
    index = np.where(y_pred == np.amax(y_pred))
    plt.title(
        "[Ground Truth] : {}\n[Model's Prediction]: {}".format(
            cifar_dict[y_test[i][0]], cifar_dict[index[1][0]]
        )
    )
    print("i = {}".format(i))
    print("Y_test value [Ground Truth] : {}".format(cifar_dict[y_test[i][0]]))
    print("Y_Pred value [Model's Prediction]: ", cifar_dict[index[1][0]])
    plt.show()

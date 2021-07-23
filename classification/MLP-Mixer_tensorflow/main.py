

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
    # Making use of keras to load the dataset. (check 'load_dataset.py')
    x_train,
    x_test,
    y_test,
    y_train
)

from utils import (
    # Importing functions and other objects from utils.py
    cifar_dict
)

from model import (
    # Function to build the model
    build_model
)

# Parameters for the model
train_bool = False    # If you want to train the model
test_display = True   # To display a random image and the prediction from the test dataset
checkpoint_path = '/home/merp/Desktop/model-zoo/classification/MLP-Mixer_tensorflow/models/cp.ckpt'
num_classes = 100          
input_shape = (32, 32, 3)   
weight_decay = 0.0001
batch_size = 128
num_epochs = 50
dropout_rate = 0.2
image_size = 64  # We'll resize input images to this size.
patch_size = 8  # Size of the patches to be extracted from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
embedding_dim = 256  # Number of hidden units.
num_blocks = 4  # Number of blocks.

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")



def train_model(model):
    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )
    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Creating a model checkpoint callback, which saves it every 30 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=30)

    #to get the latest checkpoint file
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # If we have a checkpoint file
    if not ((latest)==None):
      model.load_weights(latest)

    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr, cp_callback],
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return history to plot learning curves.
    return history

my_model = build_model()
my_model.summary()

# The below line does training otherwise
if train_bool:
    my_model = train_model(my_model)

#  To do direct testing without training
tmp_checkpoint_dir = os.path.dirname("/home/merp/Desktop/model-zoo/classification/MLP-Mixer_tensorflow/models/cp.ckpt")
latest_weights = tf.train.latest_checkpoint(tmp_checkpoint_dir)
# If we have a saved weights in train folder
if not ((latest_weights)==None):
    my_model.load_weights(latest_weights)

# To display the model's performance on a random image.
if test_display:
    import random
    i=random.randint(1,10000)
    plt.imshow(x_test[i])
    copy = x_test[i]
    copy = copy[None,:,:,:]
    y_pred = (my_model.predict(copy))
    index = np.where(y_pred == np.amax(y_pred))
    plt.title("[Ground Truth] : {}\n[Model's Prediction]: {}".format(cifar_dict[y_test[i][0]],cifar_dict[index[1][0]]))
    print("i = {}".format(i))
    print("Y_test value [Ground Truth] : {}".format(cifar_dict[y_test[i][0]]))
    print("Y_Pred value [Model's Prediction]: ",cifar_dict[index[1][0]])
    plt.show()

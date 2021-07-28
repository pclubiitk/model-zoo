import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from dataloader import load_cifar10
from models import Mobilenet_v1, depthwise_conv, standard_conv
import matplotlib.pyplot as plt

# This functions takes the arguement from the user needed to execute the training stage of the model on the cifar10 train dataset.
# The default epochs is set to 40 and the model arguement is set to mobilenetv1.

def take_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--model", type=str, default="mobilenetv1")
    args = parser.parse_args()

    num_epochs = args.epochs
    model = args.model

    return model, num_epochs


model, num_epochs = take_args()
train_ds, test_ds = load_cifar10()
input_img = Input(shape=[224, 224, 3])
results = Mobilenet_v1(input_img)
model = Model(inputs=input_img, outputs=results)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


hist = model.fit(
    train_ds, validation_data=test_ds, epochs=num_epochs, batch_size=64, verbose=1
)


# This functions plots the variation of training accuracy and validation accuracy in the first graph and then plots the variation between the training loss and 
# the validation loss in the second graph.

def plot_traingraph(hist):

    epoch = range(40)
    train_acc = hist.history["accuracy"]
    train_loss = hist.history["loss"]
    validation_acc = hist.history["val_accuracy"]
    validation_loss = hist.history["val_loss"]
    # fig ,ax= plt.subplots(nrows=1,ncols=2)

    # This plot shows the variation of training accuracy and the validation accuracy with epochs
    fig = plt.figure(figsize=(7, 5))
    plt.plot(epoch, train_acc)
    plt.plot(epoch, validation_acc)
    plt.xlabel("EPOCHS")
    plt.ylabel("ACCURACY")
    plt.title("Variation of Accuracy")
    plt.legend(["train_acc", "validation_acc"])
    plt.grid()

    # This plot shows the variation of training loss and the validation loss with epochs
    fig = plt.figure(figsize=(7, 5))
    plt.plot(epoch, train_loss)
    plt.plot(epoch, validation_loss)
    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.title("Variation of Loss")
    plt.legend(["train_loss", "validation_loss"])
    plt.grid()
    fig.tight_layout(pad=0.3)
    

model.save("mobilenetv1.h5")
print("model saved...")    

plot_traingraph(hist)

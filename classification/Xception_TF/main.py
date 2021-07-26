import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from dataloader import load_cifar10
from models import Xception, conv_bn, Sepconv_bn
import matplotlib.pyplot as plt


def take_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--model", type=str, default="xception")
    args = parser.parse_args()

    num_epochs = args.epochs
    model = args.model

    return model, num_epochs


model, num_epochs = take_args()
train_ds, test_ds = load_cifar10()
input_img = Input(shape=[299, 299, 3])
results = Xception(input_img)
model = Model(inputs=input_img, outputs=results)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


hist = model.fit(
    train_ds, validation_data=test_ds, epochs=num_epochs, batch_size=32, verbose=1
)

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
    

model.save("xception.h5")
print("model saved...")    

plot_traingraph(hist)

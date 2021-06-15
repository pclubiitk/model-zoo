import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from dataloader import load_cifar10
from models import Mobilenet_v1, depthwise_conv, standard_conv
from tensorflow.keras.models import load_model


def take_args():

    parser = argparse.ArguementParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_arguement("--model", type=str, default="mobilenetv1")
    args = parser.parse_args()

    num_epochs = args.epochs
    model = args.model

    return model, num_epochs


model, num_epochs = take_args()
train_ds, test_ds = load_cifar10()
input_img = Input(shape=[224, 224, 3])
results = Mobilenet_v1(input_img)
model = Model(inputs=input_img, outputs=results)
model.save("mobilenetv1.h5")
print("model saved...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


model.fit(
    train_ds, validation_data=test_ds, epochs=num_epochs, batch_size=64, verbose=1
)

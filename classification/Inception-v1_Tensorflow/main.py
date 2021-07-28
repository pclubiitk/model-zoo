import argparse
import tensorflow as tf
from model import inceptionv1
from dataloader import load


parser = argparse.ArgumentParser()

parser.add_argument(
    "--channels", type=int, default=10, help="Number of channels in the dataset."
)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=128)

args = parser.parse_args()


# Default dataset loaded is CIFAR-10
x_train, y_train, x_test, y_test = load()


def lr_decrease(epoch, lr):
    if epoch % 8:
        return lr
    else:
        return lr * 0.96


model = inceptionv1(args.channels)


model.compile(
    optimizer=tf.keras.optimizers.SGD(momentum=0.9),
    loss=[
        "categorical_crossentropy",
        "categorical_crossentropy",
        "categorical_crossentropy",
    ],
    loss_weights=[1.0, 0.3, 0.3],
    metrics=["accuracy"],
)


history = model.fit(
    x_train,
    y_train,
    batch_size=args.batch_size,
    epochs=args.epochs,
    validation_data=(x_test, y_test),
    callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_decrease)],
)


model.evaluate(x_test, y_test)

import argparse
import tensorflow as tf
from model import repvgg
from dataloader import load


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="A0", help="Name of the model")
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=256)

args = parser.parse_args()


arch = {
    "A0": ([1, 2, 4, 14, 1], 0.75, 2.5),
    "A1": ([1, 2, 4, 14, 1], 1, 2.5),
    "A2": ([1, 2, 4, 14, 1], 1.5, 2.75),
    "B0": ([1, 4, 6, 16, 1], 1, 2.5),
    "B1": ([1, 4, 6, 16, 1], 2, 4),
    "B2": ([1, 4, 6, 16, 1], 2.5, 5),
    "B3": ([1, 4, 6, 16, 1], 3, 5),
}


l, a, b = arch[args.model]
x_train, y_train, x_test, y_test = load()

# Class for callbacks
class mcb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


cb = mcb()


model = repvgg(a, b, l, nc=10)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=args.batch_size,
    epochs=args.epochs,
    validation_data=(x_test, y_test),
    callbacks=[cb],
)


model.evaluate(x_test, y_test)

import tensorflow as tf
import tensorflow.keras as keras
import argparse

import hyperparams as hprms
import utils
import model

parser = argparse.ArgumentParser(description='Main module to initiate training of GAN')
parser.add_argument("--epoch", default=50, help="Epochs for training. Default is 50", type=int)
parser.add_argument("--lr", default = 0.1, help="Learning Rate. Default is 0.1", type=float)
args = parser.parse_args()

epochs = args.epoch
lr = args.lr


(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train, y_train, X_test, y_test = utils.preprocess_data(X_train, y_train, X_test, y_test)
batches_train = utils.generate_random_mini_batches(X_train, y_train)

optimizer = keras.optimizers.Adam(learning_rate=lr)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
acc_metric = keras.metrics.CategoricalAccuracy()
model = model.build_model((32,32,3))

train_writer = tf.summary.create_file_writer("logs/train/")
test_writer = tf.summary.create_file_writer("logs/test/")
train_step = test_step = 0


for epoch in range(epochs):
  print(f"\nTraining on Epoch {epoch+1}")

  for batch in batches_train:

    (x_batch, y_batch) = batch

    with tf.GradientTape() as tape:
      y_preds = model(x_batch, training=True)
      loss = loss_fn(y_batch, y_preds)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    acc_metric.update_state(y_batch, y_preds)


  with train_writer.as_default():
    tf.summary.scalar("Loss", loss, step=train_step)
    tf.summary.scalar(
      "Accuracy", acc_metric.result(), step=train_step,
      )
    train_step += 1

  train_acc = acc_metric.result()
  print(f"Accuracy over epoch {epoch+1} is {train_acc}")
  acc_metric.reset_states()



batches_test = utils.generate_random_mini_batches(X_test, y_test)

for batch in batches_test:
  y_pred = model(x_batch, training=False)
  acc_metric.update_state(y_batch, y_pred)

  with test_writer.as_default():
    tf.summary.scalar("Loss", loss, step=test_step)
    tf.summary.scalar(
        "Accuracy", acc_metric.result(), step=test_step,
    )
    test_step += 1

test_acc = acc_metric.result()
print(f"Accuracy over test set is {test_acc}")
acc_metric.reset_states()
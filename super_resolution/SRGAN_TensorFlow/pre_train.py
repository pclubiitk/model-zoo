from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf
from utils import evaluate
import datetime,time

def pre_train(generator, train_dataset, valid_dataset, steps, evaluate_every=1,lr_rate=1e-4):
    loss_mean = Mean()
    pre_train_loss = MeanSquaredError()
    pre_train_optimizer = Adam(lr_rate)

    now = time.perf_counter()

    step = 0
    for lr, hr in train_dataset.take(steps):
        step = step+1

        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = generator(lr, training=True)
            loss_value = pre_train_loss(hr, sr)

        gradients = tape.gradient(loss_value, generator.trainable_variables)
        pre_train_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        loss_mean(loss_value)

        if step % evaluate_every == 0:
            loss_value = loss_mean.result()
            loss_mean.reset_states()

            psnr_value = evaluate(generator, valid_dataset)

            duration = time.perf_counter() - now
            print(
                f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

            now = time.perf_counter()

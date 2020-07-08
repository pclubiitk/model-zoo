from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf
import datetime,time
from model import vgg as VGG

# Used in content_loss
mean_squared_error = tf.keras.losses.MeanSquaredError()

# Used in generator_loss and discriminator_loss
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def generator_loss(sr_out):
    return binary_cross_entropy(tf.ones_like(sr_out), sr_out)


def discriminator_loss(hr_out, sr_out):
    hr_loss = binary_cross_entropy(tf.ones_like(hr_out), hr_out)
    sr_loss = binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
    return hr_loss + sr_loss


@tf.function
def content_loss(vgg,hr, sr):
    sr = tf.keras.applications.vgg19.preprocess_input(sr)
    hr = tf.keras.applications.vgg19.preprocess_input(hr)
    sr_features = vgg(sr) / 12.75
    hr_features = vgg(hr) / 12.75
    return mean_squared_error(hr_features, sr_features)


def train(generator, discriminator, train_ds, valid_ds, steps=2000, lr_rate=1e-4):
    generator_optimizer = Adam(learning_rate=lr_rate)
    discriminator_optimizer = Adam(learning_rate=lr_rate)
    vgg = VGG()

    pls_metric = tf.keras.metrics.Mean()
    dls_metric = tf.keras.metrics.Mean()

    steps = steps
    step = 0

    for lr, hr in train_ds.take(steps):
        step += 1

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            # Forward pass
            sr = generator(lr, training=True)
            hr_output = discriminator(hr, training=True)
            sr_output = discriminator(sr, training=True)

            # Compute losses
            con_loss = content_loss(vgg,hr, sr)
            gen_loss = generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = discriminator_loss(hr_output, sr_output)

        # Compute gradient of perceptual loss w.r.t. generator weights
        gradients_of_generator = gen_tape.gradient(
            perc_loss, generator.trainable_variables)
        # Compute gradient of discriminator loss w.r.t. discriminator weights
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        # Update weights of generator and discriminator
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

        pl, dl = perc_loss, disc_loss
        pls_metric(pl)
        dls_metric(dl)

        print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
        pls_metric.reset_states()
        dls_metric.reset_states()

    generator.save_weights('pre-trained/generator.h5')
    discriminator.save_weights('pre-trained/discriminator.h5')

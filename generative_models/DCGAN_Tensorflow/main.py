import tensorflow as tf
import numpy as np
import os
import PIL
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from MNIST_model import make_generator,make_discriminator
from utils.py import plot_loss,generate_and_save_images
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--EPOCHS', type = int, default = 50, help = "No of EPOCHS: default 50 ")
parser.add_argument('--noise_dim', type = int, default = 100, help = "Noise dimension, default 100 ")
parser.add_argument('--BATCH_SIZE', type = int, default = 128, help = "Batch size, default 128")
parser.add_argument('--num_examples_to_generate', type = int, default = 16, help = "no of images shown after each epoch in output, default 16")
parser.add_argument('--lr_gen', type = int, default = 0.0002, help = "Learning rate for generator optimizer,default 0.0002 ")
parser.add_argument('--lr_disc', type = int, default = 0.0002, help = "Learning rate for discriminator optimizer,default 0.0002 ")
parser.add_argument('--outdir', type = str, default = '.', help = "Directory in which to store data")

args = parser.parse_args()

# Loading  MNIST_Dataset
(train_images, train_labels),(_,_) = tf.keras.datasets.mnist.load_data()

BUFFER_SIZE=60000
BATCH_SIZE = args.BATCH_SIZE
EPOCHS = args.EPOCHS
noise_dim = args.noise_dim
num_examples_to_generate = args.num_examples_to_generate
lr_gen = args.lr_gen
lr_disc = args.lr_disc


seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Preparing and Normalising Dataset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Making generator and Discriminator
generator=make_generator(noise_dim)
discriminator=make_discriminator()

# Defining generator and discriminator losses
cross_entropy = BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Defining optimizers
generator_optimizer = Adam(learning_rate=lr_gen)  
discriminator_optimizer = Adam(learning_rate=lr_disc)

# Saving Checkpoints
checkpoint_dir = os.path.join(args.outdir, "training_checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Defining Training Loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss,disc_loss

def train(dataset, epochs):
  for epoch in range(epochs):
    gen_loss_list = []
    disc_loss_list = []

    for image_batch in dataset:
      t=train_step(image_batch)
      gen_loss_list.append(t[0])
      disc_loss_list.append(t[1])

    gen_loss = sum(gen_loss_list) / len(gen_loss_list)
    disc_loss = sum(disc_loss_list) / len(disc_loss_list)

    
    print (f'Epoch {epoch+1}, gen loss={gen_loss},disc loss={disc_loss}')
    
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
 
  return gen_loss_list,disc_loss_list            

# Training our model
plo=train(train_dataset, EPOCHS)

#Ploting generator and discriminator losses
plot_loss(plo[0],plo[1])

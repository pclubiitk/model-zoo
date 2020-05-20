import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os
import time
import datetime
import argparse
from tensorflow.keras import layers

print(tf.__version__)
from utils import run_from_ipython, generate_latent_points, generate_and_save_images, save_gif, generate_varying_outputs

parser = argparse.ArgumentParser()
ipython = run_from_ipython()

if ipython:
    from IPython import display

parser.add_argument('--dataset', type = str, default = "MNIST", help = "Name of dataset: MNIST (default) or CIFAR10")
parser.add_argument('--epochs', type = int, default = 0, help = "No of epochs: default 50 for MNIST, 150 for CIFAR10")
parser.add_argument('--noise_dim', type = int, default = 0, help = "No of latent Noise variables, default 62 for MNIST, 64 for CIFAR10")
parser.add_argument('--continuous_weight', type = float, default = 0.0, help = "Weight given to continuous Latent codes in loss calculation, default 0.5 for MNIST, 1 for CIFAR10")
parser.add_argument('--batch_size', type = int, default = 256, help = "Batch size, default 256")
parser.add_argument('--outdir', type = str, default = '.', help = "Directory in which to store data, don't put '/' at the end!")

args = parser.parse_args()

if args.dataset == "MNIST":
    from model_MNIST import make_generator_model, make_discriminator_model
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    if args.epochs == 0 :
        args.epochs = 50
    if args.noise_dim == 0 :
        args.noise_dim = 62
    if args.continuous_weight == 0.0:
        args.continuous_weight = 0.5
    
else :
    from model_CIFAR10 import make_generator_model, make_discriminator_model
    (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
    if args.epochs == 0 :
        args.epochs = 150
    if args.noise_dim == 0 :
        args.noise_dim = 64
    if args.continuous_weight == 0.0:
        args.continuous_weight = 1
            
if not os.path.exists(f"{args.outdir}/assets/{args.dataset}"):
    os.makedirs(f"{args.outdir}/assets/{args.dataset}")

#normalizing the images
train_images = (train_images - 127.5) / 127.5

##### DEFINE GLOBAL VARIABLES AND OBJECTS ######
BUFFER_SIZE = 600000
BATCH_SIZE = args.batch_size
epochs = args.epochs
noise_dim = args.noise_dim
continuous_dim = 2
categorical_dim = 10
num_examples_to_generate = 100
continuous_weight = args.continuous_weight
seed, _, _ = generate_latent_points(num_examples_to_generate, noise_dim, categorical_dim, continuous_dim) # A constant sample of latent points so as to create images

 # Define Generator
generator = make_generator_model(noise_dim)
print("\nGenerator : ")
print(generator.summary())
discriminator = make_discriminator_model()
print("\nDiscriminator : ")
print(discriminator.summary())

print("Dataset : ", args.dataset)
###########################################

# Converting data to tf Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# defining losses
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

#defining optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5 )
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#defining storage points for checkpoints
checkpoint_dir = f'{args.outdir}/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer =discriminator_optimizer,generator=generator,discriminator=discriminator)

#defining loss metrics for Plotting purposes with tensorboard
discriminator_loss_metric = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)
discriminator_real_accuracy_metric = tf.keras.metrics.BinaryCrossentropy('discriminator_real_accuracy', from_logits=True)
discriminator_fake_accuracy_metric = tf.keras.metrics.BinaryCrossentropy('discriminator_fake_accuracy', from_logits=True)
generator_loss_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
categorical_loss_metric = tf.keras.metrics.Mean('categorical_loss', dtype=tf.float32)
continuous_loss_metric = tf.keras.metrics.Mean('continuous_loss', dtype=tf.float32)

# Save points for metrics
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base = f"{args.outdir}/logs/gradientTape/{current_time}"
disc_log_dir = base + '/discriminator'
gen_log_dir = base + '/generator'
cont_log_dir = base + '/cont'
cat_log_dir = base + '/cat'

# Create summary writers
disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
cat_summary_writer = tf.summary.create_file_writer(cont_log_dir)
cont_summary_writer = tf.summary.create_file_writer(cat_log_dir)

##################################
# A train step to train the model on a minibatch

def train_step(images):
    noise, categorical_input, continuous_input = generate_latent_points(BATCH_SIZE, noise_dim, categorical_dim, continuous_dim)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      disc_loss, real_loss, fake_loss, categorical_loss, continuous_loss = discriminator_loss(real_output, fake_output, categorical_input, continuous_input)
      gen_loss = generator_loss(fake_output, categorical_loss, continuous_loss)

    discriminator_loss_metric(disc_loss)
    generator_loss_metric(gen_loss)
    discriminator_real_accuracy_metric(tf.ones_like(real_output[:,0]), real_output[:,0])
    discriminator_fake_accuracy_metric(tf.zeros_like(fake_output[:,0]), fake_output[:,0])
    categorical_loss_metric(categorical_loss)
    continuous_loss_metric(continuous_loss)

    print(f"Losses - Disc : [{disc_loss}], Gen : [{gen_loss}], \n categorical loss : {categorical_loss}, continuous loss : {continuous_loss}")
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
####################################

def discriminator_loss(real_output, fake_output, categorical_input, continuous_input):
    real_loss = binary_cross_entropy(tf.ones_like(real_output[:,0]), real_output[:,0])
    fake_loss = binary_cross_entropy(tf.zeros_like(fake_output[:,0]), fake_output[:,0])
    
    categorical_output = fake_output[:,1:1 + categorical_dim]
    continuous_output = fake_output[:, 1+categorical_dim : ]

    categorical_loss = categorical_cross_entropy(categorical_input, categorical_output)
    continuous_loss = tf.reduce_mean((2*(continuous_output - continuous_input))**2)
    
    total_loss = real_loss + fake_loss + continuous_weight*continuous_loss + categorical_loss
    return total_loss, real_loss, fake_loss, categorical_loss, continuous_loss

#####################################

def generator_loss(fake_output, categorical_loss, continuous_loss):
    gen_loss = binary_cross_entropy(tf.ones_like(fake_output[:,0]), fake_output[:,0])
    return gen_loss +  continuous_weight*continuous_loss + categorical_loss
    
#####################################

def main():
    # begin the training loop
    
    for epoch in range(epochs):
        start = time.time()
        print(f"EPOCH : {epoch+1}")
        for image_batch in train_dataset:
            train_step(image_batch)
        # Produce images for the GIF
        if ipython:
            display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed, outdir = args.outdir, dataset = args.dataset)
    
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)
        
        # writing to summary writers
        with disc_summary_writer.as_default():
          tf.summary.scalar('Loss', discriminator_loss_metric.result(), step = epoch)
          tf.summary.scalar('Real Accuracy', discriminator_real_accuracy_metric.result(), step = epoch)
          tf.summary.scalar('Fake Accuracy', discriminator_fake_accuracy_metric.result(), step = epoch)
        
        with cat_summary_writer.as_default():
          tf.summary.scalar('Loss', categorical_loss_metric.result(), step = epoch)
        
        with cont_summary_writer.as_default():
          tf.summary.scalar('Loss', continuous_loss_metric.result(), step = epoch)
        
        with gen_summary_writer.as_default():
          tf.summary.scalar('Loss', generator_loss_metric.result(), step = epoch)
    
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print(f'Epoch results: Discriminator Loss: {discriminator_loss_metric.result()}, Real Accuracy: {discriminator_real_accuracy_metric.result()}, Fake Accuracy: {discriminator_fake_accuracy_metric.result()}')
        print(f'               Generator Loss: {generator_loss_metric.result()}')
    
        discriminator_loss_metric.reset_states()
        discriminator_real_accuracy_metric.reset_states()
        discriminator_fake_accuracy_metric.reset_states()
        generator_loss_metric.reset_states()
        categorical_loss_metric.reset_states()
        continuous_loss_metric.reset_states()
    
    # Generate after the final epoch
    if ipython:
        display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed, outdir = args.outdir, dataset = args.dataset)
                             
    save_gif(args.outdir, args.dataset)
    
    # For producing outputs with constant noise and varying continuous and categorical latent codes
    
    generate_varying_outputs(generator, num_examples_to_generate, noise_dim, args.dataset, args.outdir)
    
if __name__ == '__main__':
    main()

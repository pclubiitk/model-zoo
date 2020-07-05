import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from models import make_discriminator, make_generator
from utils import plot_images, make_gif
from dataloader import dataloader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--outdir', type=str, required=True,default='./cgan/')
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--latent_size', type=int, default=100)

    args = parser.parse_args()

    return args

args = parse_args()

epochs = args.epochs
dataset = args.dataset
batch_size = args.batch_size
outdir = os.path.join(args.outdir, dataset)
lr = args.learning_rate
latent_size = args.latent_size

if not os.path.exists(outdir):
    os.makedirs(outdir)

def train(dataset, latent_size, batch_size, epochs, lr, outdir, decay = 6e-8):
    
    x_train, y_train, image_size, num_labels = dataloader(dataset)
    model_name = "cgan_" + dataset
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels, )

    ##################################################################

    inputs = layers.Input(shape=input_shape, name='discriminator_input')
    labels = layers.Input(shape=label_shape, name='class_labels')

    discriminator = make_discriminator(inputs, labels, image_size)

    optimizer = keras.optimizers.Adam(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    ##################################################################

    input_shape = (latent_size, )
    inputs = layers.Input(shape=input_shape, name='z_input')
    generator = make_generator(inputs, labels, image_size)
    generator.summary()

    optimizer = keras.optimizers.Adam(lr=lr*0.5, decay=decay*0.5)
    
    discriminator.trainable = False

    outputs = discriminator([generator([inputs, labels]), labels])
    gan = keras.models.Model([inputs, labels],
                        outputs,
                        name=model_name)
    gan.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    gan.summary()

    ##################################################################

    noise_input = np.random.uniform(-1.0, 1.0, size=[100, latent_size])
    noise_class = np.eye(num_labels)[np.arange(0, 100) % num_labels]
    train_size = x_train.shape[0]
    batch_count = int(train_size / batch_size)
    G_Losses = []
    D_Losses = []

    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_class, axis=1))

    for e in range(1, epochs+1):
        for b_c in range(batch_count):
            random = np.random.randint(0, train_size, size=batch_size)
            real_images = x_train[random]
            real_labels = y_train[random]

            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
            fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
            fake_images = generator.predict([noise, fake_labels])

            x = np.concatenate((real_images, fake_images))
            labels = np.concatenate((real_labels, fake_labels))

            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0.0

            d_loss, d_acc = discriminator.train_on_batch([x, labels], y)
            log = "[discriminator loss: %f || acc: %f || ]" % (d_loss, d_acc)

            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
            fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
            y = np.ones([batch_size, 1])

            gan_loss, gan_acc = gan.train_on_batch([noise, fake_labels], y)
            log = "%s [gan loss: %f || acc: %f]" % (log, gan_loss, gan_acc)

        print("[epoch: %d] %s" % (e, log))
          
        G_Losses.append(gan_loss)
        D_Losses.append(d_loss)
        
        plot_images(generator,
                    noise_input=noise_input,
                    noise_class=noise_class,
                    outdir = outdir,
                    show=True,
                    epoch=e,)
          
    plt.plot(G_Losses, label='Generator')
    plt.plot(D_Losses, label='Discriminator')
    plt.legend()
    plt.savefig(outdir + "plot.png")
    plt.show()

    make_gif(outdir, model_name, epochs)

train(dataset = dataset, latent_size = latent_size, batch_size = batch_size, epochs = epochs, lr = lr, outdir = outdir, decay = 6e-8)

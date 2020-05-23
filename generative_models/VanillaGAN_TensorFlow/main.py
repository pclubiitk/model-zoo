import os
import argparse
import numpy as np

from dataloader import load_data
from utils import plot_generated_images,make_gif
from models import create_generator,create_gan,create_discriminator

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

ipython = run_from_ipython()

if ipython:
    from IPython import display

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--outdir', type=str, required=True,default='.')
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--encoding_dims', type=int, required=True,default=100)

args = parser.parse_args()

outdir = args.outdir
if not os.path.exists(outdir):
    os.makedirs(outdir)

epochs = args.epochs
batch_size = args.batch_size
outdir = args.outdir
learning_rate = args.learning_rate
beta_1 = args.beta_1
encoding_dims = args.encoding_dims

def training(epochs, batch_size):

    X_train = load_data()
    batch_count = int(X_train.shape[0] / batch_size)

    generator= create_generator(learning_rate,beta_1,encoding_dims)
    discriminator= create_discriminator(learning_rate,beta_1)
    gan = create_gan(discriminator, generator,encoding_dims)

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    seed = np.random.normal(0,1, [25, encoding_dims])

    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
        for _ in range(batch_count):

          noise= np.random.normal(0,1, [batch_size, encoding_dims])
          generated_images = generator.predict(noise)

          image_batch = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]

          discriminator.trainable=True
          d_loss_real = discriminator.train_on_batch(image_batch, valid)
          d_loss_fake = discriminator.train_on_batch(generated_images, fake)
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

          noise= np.random.normal(0,1, [batch_size, encoding_dims])

          discriminator.trainable=False
          g_loss = gan.train_on_batch(noise,valid)

          print ("%d [D loss: %f] [G loss: %f]" % (e, d_loss, g_loss))
        if ipython:
            display.clear_output(wait=True)
        plot_generated_images(e, generator,seed,outdir)
    generator.save('{}/gan_model'.format(outdir))

training(epochs,batch_size)

make_gif(outdir)

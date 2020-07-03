import tensorflow as tf
import argparse
import datetime
import os
from scipy.io.wavfile import write as write_wav
from model import Generator, Discrimiator
from utils import generator_loss, discriminator_loss
from wgan_gp import WGAN
from dataset import get_dataset

if __name__ == "__main__":
    # Argument Parser Stuff
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                        help="Train WaveGAN model", action="store_true")
    parser.add_argument("--use_tensorboard",
                        help="use this to use tensorboard while training", action="store_true")
    parser.add_argument("--print_summary",
                        help="use this to print model summary before training", action="store_true")
    parser.add_argument("--save_model",
                        help="use this to save the models after training", action="store_true")
    parser.add_argument("--generate",
                        help="Generate a random Sample as 'output.wav'", action="store_true")
    parser.add_argument('--latent_dim', type=int, default=100,
                        help="Dimentions of the Latent vector used for generating samples. Default: 100")
    parser.add_argument('--epochs', type=int, default=50,
                        help="No of epochs for training: default 50 ")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size to use while training. paper suggests 64. Default: 64")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate for training,Default: 1e-4 ")
    parser.add_argument('--beta1', type=float, default=0.5,
                        help="We are using Adam optimizer, as suggested in the paper. this is the beta 1 paprameter for the Adam optimizer. Default: 0.5")
    parser.add_argument('--beta2', type=float, default=0.9,
                        help="We are using Adam optimizer, as suggested in the paper. this is the beta 2 paprameter for the Adam optimizer. Default: 0.9")
    parser.add_argument('--d_per_g', type=int, default=5,
                        help="No. of updates discriminator per generator update. Default: 5 ")
    parser.add_argument('--gp_weight', type=int, default=10,
                        help="GP Weight for Wgan-GP (lambda). Default: 10")
    args = parser.parse_args()

    # training the model
    if args.train:
        # Hyper parameters taken as arguments
        batch_size = args.epochs
        lr = args.lr
        beta1 = args.beta1
        beta2 = args.beta2
        latent_dim = args.latent_dim
        d_per_g = args.d_per_g
        gp_weight = args.gp_weight
        epochs = args.epochs

        # Create Data Pipe
        train_ds = get_dataset('train', batch_size=batch_size)

        # Get both models
        generator = Generator()
        discriminator = Discrimiator()

        # print summary of the models
        if args.print_summary:
            generator.summary()
            discriminator.summary()

        # Specifying the Optimizer to use while training
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        g_optimizer = tf.keras.optimizers.Adam(lr, beta1, beta2)
        d_optimizer = tf.keras.optimizers.Adam(lr, beta1, beta2)

        # Callbacks
        if args.use_tensorboard:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)

        # Get the wgan model
        wgan = WGAN(
            discriminator=discriminator,
            generator=generator,
            latent_dim=latent_dim,
            discriminator_extra_steps=d_per_g,
            gp_weight=gp_weight
        )

        # Compile the wgan model
        wgan.compile(
            d_optimizer=d_optimizer,
            g_optimizer=g_optimizer,
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss,
        )

        # Start training
        if args.use_tensorboard:
            wgan.fit(train_ds, batch_size=batch_size, epochs=epochs,
                     callbacks=[tensorboard_callback])
        else:
            wgan.fit(train_ds, batch_size=batch_size, epochs=epochs)

        # Saving the models
        if args.save_model:
            os.mkdir('trained')
            generator.save_weights('trained/g.h5')
            discriminator.save_weights('trained/d.h5')

    # Generating a Sample
    if args.generate:
        if not os.path.exists('trained'):
            print('unable to generate samples. No trained model exists')
        else:
            print("Generating a random Sample of audio as 'output.wav'")
            if not args.train:
                generator = Generator()
                discriminator = Discrimiator()
                try:
                    generator.load_weights('trained/g.h5')
                    discriminator.load_weights('trained/d.h5')
                except:
                    print("An Error occured while loading the model")
            noise = tf.random.normal([1, 100])
            generated = generator(noise, training=False)
            data = generated[0, :, :, :].reshape(128*128*1)
            write_wav('output.wav', 16000, data)

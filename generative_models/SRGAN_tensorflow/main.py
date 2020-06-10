from srgan.model import generator, discriminator
from srgan.dataset import DIV2K
from srgan.pre_train import pre_train
from srgan.train import train
import cv2
import argparse
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--STEPS', type=int, default=2000,
                    help="No of steps for training: default 2000 ")
parser.add_argument('--PREGENSTEPS', type=int, default=1000,
                    help="No of steps for generator pre training: default 1000 ")
parser.add_argument('--BATCH_SIZE', type=int, default=128,
                    help="Batch size, default 128")
parser.add_argument('--lr_pre_gen', type=int, default=1e-4,
                    help="Learning rate for generator pre training,default 1e-4 ")
parser.add_argument('--lr_gen', type=int, default=1e-4,
                    help="Learning rate for GAN training,default 1e-4 ")
parser.add_argument(
    "--pretrained", help="converts the input.jpg to output.jpg using the pretrained model makin it 4x larger.", action="store_true")

args = parser.parse_args()

if args.pretrained:
    img = cv2.imread('input.jpg')
    generator = generator()
    generator.load_weights('pre-trained/generator.h5')
    x,y,c = img.shape
    img = generator.predict(tf.reshape(img,[1,x,y,c]))
    cv2.imwrite('output.jpg', img[0])

else:
    train_loader = DIV2K(scale=4, subset='train')
    train_ds = train_loader.dataset(
        batch_size=args.BATCH_SIZE, random_transform=True, repeat_count=None)
    valid_loader = DIV2K(scale=4, subset='valid')
    valid_ds = valid_loader.dataset(
        batch_size=1, random_transform=False, repeat_count=1)

    generator = generator()
    discriminator = discriminator()

    pre_train(train_ds, valid_ds, steps=args.PREGENSTEPS,
              evaluate_every=1, lr_rate=args.le_pre_gen)

    train(generator, discriminator, train_ds, valid_ds,
          steps=args.STEPS, lr_rate=args.lr_gen)


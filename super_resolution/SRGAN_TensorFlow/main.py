from model import generator, discriminator
from dataset import DIV2K
from pre_train import pre_train
from train import train
import tensorflow as tf
import cv2
import argparse
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--STEPS', type=int, default=2000,
                    help="No of steps for training: default 2000 ")
parser.add_argument('--HR_PATCH_SIZE', type=int, default=96,
                    help="Size of the high resolution patch to use in training. Default: 96")
parser.add_argument('--UPSCALING', type=int, default=4,
                    help="How many times do you need to upscale the image. Default: 4")
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
    train_loader = DIV2K(scale=args.UPSCALING, subset='train',HR_SIZE=args.HR_PATCH_SIZE)
    train_ds = train_loader.dataset(
        batch_size=args.BATCH_SIZE, random_transform=True, repeat_count=None)
    valid_loader = DIV2K(scale=args.UPSCALING, subset='valid',HR_SIZE=args.HR_PATCH_SIZE)
    valid_ds = valid_loader.dataset(
        batch_size=1, random_transform=False, repeat_count=1)

    generator = generator()
    discriminator = discriminator(HR_SIZE=args.HR_PATCH_SIZE)

    pre_train(generator,train_ds, valid_ds, steps=args.PREGENSTEPS,
              evaluate_every=1, lr_rate=args.lr_pre_gen)

    train(generator, discriminator, train_ds, valid_ds,
          steps=args.STEPS, lr_rate=args.lr_gen)


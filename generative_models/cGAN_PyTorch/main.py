import os
import time
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision.utils import save_image
torch.backends.cudnn.benchmark = True
import matplotlib.pyplot as plt
from dataset import get_loader
import imageio
from model import *


def main():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=100)
    # run config
    parser.add_argument('--outdir', type=str, default='./result/')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ndata', type=str, required=True)

    # optim config
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    args.channel = 3 if args.ndata=='cifar10' else 1

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        os.makedirs(outdir+'generator_state/')
        os.makedirs(outdir+'discriminator_state/')
        os.makedirs(outdir+'Image/')
        
    dataloader = get_loader(args.batch_size, args.num_workers, args.image_size, args.ndata)
    
    generator = Generator(args).to(args.device)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    discriminator = Discriminator(args).to(args.device)
    discriminator.apply(init_weights)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    loss = torch.nn.BCELoss()

    label_type = torch.LongTensor
    img_type = torch.FloatTensor
    
    # Sample noise
    fix_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (args.num_class ** 2, args.latent_dim)))).to('cuda:0')
    
    # Get labels ranging from 0 to n_classes for n rows
    fix_labels = np.array([num for _ in range(args.num_class) for num in range(args.num_class)])
    fix_labels = Variable(torch.LongTensor(fix_labels)).to('cuda:0')

    G_losses, D_losses = train(args,generator,discriminator,dataloader,loss, img_type, label_type,gen_optimizer,d_optimizer,fix_labels,fix_z)

    # Plotting the loss graph    
    plt.plot(G_losses, label='Generator')
    plt.plot(D_losses, label='Discriminator')
    plt.legend()
    plt.savefig(args.outdir+"plot.png")
    plt.show()

    # Making the GIF
    image = []
    for i in range(1,args.epochs+1):
      image.append(imageio.imread(args.outdir+'Image/'+str(i)+'.png'))
    imageio.mimsave(args.ndata+'.gif', image, fps=5)


def sample_image(args, z, labels, batches_done, generator):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, args.outdir + "Image/%d.png" % batches_done, nrow=args.num_class, normalize=True)


def train(args, generator, discriminator, dataloader, loss, img_type, label_type, gen_optimizer, d_optimizer, fix_label, fix_noise):

    generator.train()
    discriminator.train()
    G_losses = []
    D_losses = []

    for epoch in range(1, args.epochs + 1):
        G_loss = 0.
        D_loss = 0.
        start_time = time.time()
        for i, data in enumerate(dataloader):
            (imgs, labels) = data

            batch_size = imgs.shape[0]
            # print(batch_size)
            imgs = Variable(imgs.type(img_type)).to(args.device)
            labels = Variable(labels.type(label_type)).to(args.device)

            # Creating real and fake label for calculation of loss
            r_label = Variable(img_type(batch_size, 1).fill_(0.9)).to(args.device)
            f_label = Variable(img_type(batch_size, 1).fill_(0.0)).to(args.device)

            # Training Generator

            gen_optimizer.zero_grad()

            noise = Variable(img_type(np.random.normal(0, 1, (batch_size, args.latent_dim)))).to(args.device)
            rand_label = Variable(label_type(np.random.randint(0, args.num_class, batch_size))).to(args.device)
            dis = discriminator(generator(noise, rand_label), rand_label)
            # print(type(dis),'  ',type(r_label))
            g_loss = loss(dis, r_label)
            g_loss.backward()
            gen_optimizer.step()

            # Training Discriminator

            d_optimizer.zero_grad()

            noise = Variable(img_type(np.random.normal(0, 1, (batch_size, args.latent_dim)))).to(args.device)
            rand_label = Variable(label_type(np.random.randint(0, args.num_class, batch_size))).to(args.device)

            d_real = discriminator(imgs, labels)
            loss_real = loss(d_real, r_label)

            d_fake = discriminator(generator(noise, rand_label).detach(), rand_label)
            # print(d_fake.shape," ",f_label.shape)
            loss_fake = loss(d_fake, f_label)

            d_loss = loss_fake + loss_real

            d_loss.backward()
            d_optimizer.step()

            G_loss += g_loss.item()
            D_loss += d_loss.item()

        print('Epoch {} || G_loss: {} || D_loss: {} || Time elapsed: {}'.format(epoch, G_loss / (i), D_loss / (i),
                                                                                time.time() - start_time))
        G_losses.append(G_loss / (i))
        D_losses.append(D_loss / (i))

        sample_image(args, fix_noise, fix_label, epoch, generator)

        # Checkpoint
        torch.save(generator.state_dict(), args.outdir + 'generator_state/generator_{}_.pth'.format(epoch))
        torch.save(discriminator.state_dict(), args.outdir + 'discriminator_state/discriminator_{}_.pth'.format(epoch))

    return G_losses, D_losses

if __name__ == '__main__':
    main()
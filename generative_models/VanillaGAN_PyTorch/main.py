#import base libraries
import os
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random
import time
from tqdm.notebook import tqdm
# nn libraries
import torch.nn as nn
from torch.autograd.variable import Variable
import torch
import torchvision
#import written libraries
from  eval  import make_gif,plot_loss,show_generator
from dataloader import get_loader
from models import Generator,Discriminator

def parse_args():
    parser = argparse.ArgumentParser()

    # Model config
    parser.add_argument('--latent_dim', type=int, default=100)
    # optim config
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--milestones', type=str, default='[80, 120]')
    parser.add_argument('--lr_decay', type=float, default=0.1)
    #run_config
    parser.add_argument('--device', type=str,default='cpu')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--k_steps', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=64)
    parser.add_argument('--outdir', type=str,default='./results/')
    
    args = parser.parse_args()

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('milestones', json.loads(args.milestones)),
        ('lr_decay', args.lr_decay),
    ])

    run_config = OrderedDict([
        ('device', args.device),
        ('num_workers', args.num_workers),
        ('k_steps',args.k_steps),
        ('test_size',args.test_size),
        ('outdir',args.outdir),
    ])

    data_config = OrderedDict([
        ('Dataset','MNIST'),
        ])

    model_config = OrderedDict([
        ('latent_dim',args.latent_dim),
        ])
    
    config = OrderedDict([
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
        ('model_config', model_config),
    ])

    return config

config = parse_args()

run_config = config['run_config']
optim_config = config['optim_config']
device = run_config['device']
model_config = config['model_config']

#create out directory
outdir = run_config['outdir']
if not os.path.exists(outdir):
    os.makedirs(outdir)

outpath = os.path.join(outdir, 'config.json')
with open(outpath, 'w') as fout:
    json.dump(config, fout, indent=2)

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data.to(device)

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data.to(device)

def noise(size):
  n = Variable(torch.randn(size, model_config['latent_dim']))
  return n.to(device)    

def train_discriminator(optimizer,real_data,fake_data):

  optimizer.zero_grad()
  prediction_real = discriminator(real_data)
  loss_real = criterion(prediction_real,real_data_target(real_data.size(0)))
  loss_real.backward()

  prediction_fake = discriminator(fake_data)
  loss_fake = criterion(prediction_fake,fake_data_target(fake_data.size(0)))
  loss_fake.backward()

  optimizer.step()
  return loss_real+loss_fake

def train_generator(optimizer,fake_data):

  optimizer.zero_grad()
  prediction = discriminator(fake_data)
  error = criterion(prediction,real_data_target(prediction.size(0)))
  error.backward()
  optimizer.step()

  return error

def train(data_loader,discriminator,generator,epochs,criterion,d_optimizer,g_optimizer,test_noise):

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
  
        start_time = time.time()
        g_error=0.0
        d_error=0.0
        #progress bar
        t = tqdm(data_loader, desc='epoch:{} loss:{:.4f} accuracy:{}'.format(epoch, 0.0, 'NA'), leave=True)

        for i,data in enumerate(t):

            imgs,_ = data
            n = len(imgs)

            #train discrminator
            for j in range(d_steps):

                fake_data = generator(noise(n)).detach()
                real_data = imgs.to(device)
                d_error+=train_discriminator(d_optimizer, real_data, fake_data)

            #train generator
            fake_data = generator(noise(n))
            g_error+=train_generator(g_optimizer,fake_data)
        #store images
        img = generator(test_noise).reshape(-1,1,28,28).cpu().detach()
        img_grid = torchvision.utils.make_grid(img)
        images.append(img_grid)
        #store mean loss
        g_losses.append(g_error/i)  
        d_losses.append(d_error/i) 
        
        print(f'Epoch {epoch}: g_loss: {(g_error/i):.8f} d_loss: {(d_error/i):.8f} time:{time.time()-start_time} seconds')      

data_loader = get_loader(optim_config['batch_size'],run_config['num_workers'])
generator = Generator(model_config['latent_dim']).to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=optim_config['base_lr'],weight_decay=optim_config['weight_decay'])
g_optimizer = torch.optim.Adam(generator.parameters(), lr=optim_config['base_lr'],weight_decay=optim_config['weight_decay'])
g_losses = []
d_losses = []
images = []
criterion = nn.BCELoss()
d_steps = run_config['k_steps']
epochs = optim_config['epochs']
test_size = run_config['test_size']
test_noise = noise(test_size)

train(data_loader,discriminator,generator,epochs,criterion,d_optimizer,g_optimizer,test_noise)

plot_loss(g_losses,d_losses,outdir)
make_gif(generator,images,outdir)
show_generator(generator,test_noise)

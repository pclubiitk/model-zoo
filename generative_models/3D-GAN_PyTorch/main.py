import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

import os
import time
from collections import OrderedDict
import logging
import argparse

from dataloader import ModelNet10GAN
from models import generator, discriminator
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

parser=argparse.ArgumentParser()

parser.add_argument('--latent_dim', type=int, default=200)
parser.add_argument('--directory', type=str, default='./')

parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--gen-lr', type=float, default=0.0025)
parser.add_argument('--dis-lr', type=float, default=0.00001)
parser.add_argument('--threshold', type=float, default=0.8)

parser.add_argument('--filename', type=str, default='monitor.npy.gz')
parser.add_argument('--batch_size', type=int, default=50)
args = parser.parse_args()

# Defining ModelNet10 dataset for GAN
dataset=ModelNet10GAN(filename=args.filename,dir=args.directory)

# loading dataset into Dataloader
data_loader=loader.DataLoader(dataset, batch_size=args.batch_size)

num_epochs=args.epochs
optimizerD=optim.Adam(D_.parameters(),lr=args.dis-lr,betas=(0.5,0.999))
optimizerG=optim.Adam(G_.parameters(),lr=args.gen-lr,betas=(0.5,0.999))

G_=generator().to(device)
D_=discriminator().to(device)

# Lists to store d_losses and g_losses.
G_losses=[]
D_losses=[]
iters = 0
print("Starting training loop...")

for epoch in range(num_epochs):

    # For each batch in epoch.
    
    RunningLossG=0
    RunningLossD=0

    for i,data in enumerate(data_loader,1):
        optimizerG.zero_grad(), G_.zero_grad()
        optimizerD.zero_grad(), D_.zero_grad()
        bSize=data.size(0)

        # Train D

        real_data=data.to(device)
        noise=torch.normal(torch.zeros(bSize, 200), 
                             torch.ones(bSize, 200) * .33).to(device)
        fake_data=G_(noise)

        dReal=D_(real_data)
        dFake=D_(fake_data)

        d_loss=-torch.mean(torch.log(dReal)+torch.log(1-dFake))

        d_accuracy=((dReal>=0.5).float().mean()+(dFake<0.5).float().mean())/2
        g_accuracy=(dFake>=0.5).float().mean()

        train_dis=d_accuracy<0.8

        if train_dis:
            D_.zero_grad()
            d_loss.backward()
            optimizerD.step()

        # Train G

        noise = torch.normal(torch.zeros(bSize, 200),
                             torch.ones(bSize, 200) * .33).to(device)
        fake_data=G_(noise)
        dFake=D_(fake_data)
        g_loss=-torch.mean(torch.log(dFake))

        D_.zero_grad()
        G_.zero_grad()
        g_loss.backward()
        optimizerG.step()

        RunningLossD+=d_loss.item()
        RunningLossG+=g_loss.item()

        if (i+1)%5==0 :
            print('[%d/%d] D-Loss: %.4f G-Loss: %.4f dis: %.4f gen: %.4f'%(epoch+1,num_epochs,d_loss.item(),g_loss.item(),d_accuracy.item(),g_accuracy.item()))

    G_losses.append(RunningLossG)
    D_losses.append(RunningLossD)

directory=args.directory
threshold=args.threshold

torch.save(G_.state_dict(),'./G_{0}.pth'.format(args.epochs))
torch.save(D_.state_dict(),'./D_{0}.pth'.format(args.epochs))

voxel_plot(directory=directory, threshold=threshold)
loss_plot(G_losses=G_losses, D_losses=D_losses)


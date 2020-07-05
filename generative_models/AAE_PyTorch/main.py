import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as loader
from models import *
from utils import *
import warnings

device="cuda:0" if torch.cuda.is_available() else "cpu"

import os
import time
from collections import OrderedDict
import logging
import argparse
import warnings

parser=argparse.ArgumentParser()

parser.add_argument('--directory', help='directory of dataset', type=str, default='./')
parser.add_argument('--epochs', help='total number of epochs you want to run. Default: 100', type=int, default=100)
parser.add_argument('--batch_size', help='Batch size for dataset', type=int, default=16)
parser.add_argument('--gen_lr', help='generator learning rate', type=float, default=6e-4)
parser.add_argument('--dis_lr', help='discriminator learning rate', type=float, default=8e-4)
parser.add_argument('--download', help='Argument to download dataset. Set to True.', type=bool, default=True)

args = parser.parse_args()

transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,),(0.5,)),
])
dataset=torchvision.datasets.MNIST(root=args.directory,train=True,transform=transform,download=args.download)
data_loader=loader.DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

enc=models.encoder().to(device)
dec=models.decoder().to(device)
D_=models.discriminator().to(device)

op_enc=optim.Adam(enc.parameters(),lr=args.gen_lr)
op_dec=optim.Adam(dec.parameters(),lr=args.gen_lr)
op_gen=optim.Adam(enc.parameters(),lr=args.dis_lr)
op_disc=optim.Adam(D_.parameters(),lr=args.dis_lr)

warnings.filterwarnings("ignore")

num_epochs=args.epochs

recloss=[]
dloss=[]
gloss=[]
TINY=1e-8
for epoch in range(num_epochs):
    reconst_loss=.0
    dis_loss=.0
    gent_loss=.0
    start=time.time()
    for i,data in enumerate(data_loader):
        enc.train()
        dec.train()
        D_.train()

        # Updating autoencoder network
        op_enc.zero_grad(),op_dec.zero_grad()
        data=data[0].to(device) # We only need images
        bsize=data.size(0)
        z_gen=enc(data)
        out=dec(z_gen)
        # out=out.view(bsize,1,28,28)
        # recon=recon_loss(out,data)
        recon=F.binary_cross_entropy(out.view(bsize,-1)+TINY,data.view(bsize,-1)+TINY)
        recon.backward()
        op_enc.step()
        op_dec.step()
        reconst_loss+=recon.item()

        # Updating discriminator
        enc.eval()
        op_disc.zero_grad()
        z_real=(torch.randn(bsize,8)*5).to(device).requires_grad_(True) # Sample from N(0,5)
        z_gen=enc(data)
        D_real,D_gen=D_(z_real),D_(z_gen)
        # D_loss=disc_loss(D_real,torch.ones((bsize,1)).to(device)) + disc_loss(D_gen,torch.zeros((bsize,1)).to(device))
        D_loss=-torch.mean(torch.log(D_real+TINY)+torch.log(1-D_gen+TINY))
        D_loss.backward()
        op_disc.step()
        dis_loss+=D_loss.item()

        # Updating generator (encoder)
        enc.train()
        op_gen.zero_grad()
        D_.eval()
        z_gen=enc(data)
        D_gen=D_(z_gen)
        # g_loss=gen_loss(D_gen,torch.ones((bsize,1)).to(device))
        g_loss=-torch.mean(torch.log(D_gen+TINY))
        g_loss.backward()
        op_gen.step()
        gent_loss+=g_loss.item()

    print("[%d/%d] recon_loss: %.4f dis_loss: %.4f gen_loss: %.4f time elapsed: %.4f"%(epoch+1,num_epochs,reconst_loss,dis_loss,gent_loss,time.time()-start))
    recloss.append(reconst_loss)
    dloss.append(dis_loss)
    gloss.append(gent_loss)

utils.plot_random() # Plots a randomly generated character

utils.plot_losses(recloss,dloss,gloss)# Plot losses

# utils.interpolate_characters(s1,s2,'./interpolated')


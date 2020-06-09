import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import CelebA
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as loader
# from torchsummary import summary
import tqdm
from models import Discriminator, Generator
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import time
from collections import OrderedDict
import logging
import argparse
import warnings


parser=argparse.ArgumentParser()

parser.add_argument('--directory', help='directory of dataset', type=str, default='./')
parser.add_argument('--epochs', help='total number of epochs you want to run. Default: 20', type=int, default=20)
parser.add_argument('--batch_size', help='Batch size for dataset', type=int, default=16)
parser.add_argument('--gen_lr', help='generator learning rate', type=float, default=1e-4)
parser.add_argument('--dis_lr', help='discriminator learning rate', type=float, default=1e-4)
parser.add_argument('--d_times', help='No of times you want D to update before updating G', type=int, default=5)
parser.add_argument('--lam_cls', help='Value of lambda for domain classification loss', type=int, default=1)
parser.add_argument('--lam_recomb', help='Value of lambda for image recombination loss', type=int, default=10)
parser.add_argument('--image_dim', help='Image dimension you want to resize to.', type=int, default=64)
parser.add_argument('--download', help='Argument to download dataset. Set to True.', type=bool, default=True)
parser.add_argument('--eval_idx', help='Index of image you want to run evaluation on.', type=int, default=0)
# parser.add_argument('--eval_attr', '--list', nargs='+', help='Attributes you want to translate the eval image to.',
#                         default=[0,0,1,0,1])
parser.add_argument('--attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset', 
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

args = parser.parse_args()

c_dims=len(args.selected_attrs)

transform=transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.CenterCrop(178),
    transforms.Resize(size=64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])
dataset=CelebA(root=args.directory ,attributes=args.selected_attrs,transform=transform,download=args.download)

data_loader=loader.DataLoader(dataset,batch_size=args.batchsize)

def fakeLabels(lth):
    """
    lth (int): no of labels required
    """
    label=torch.tensor([])
    for i in range(lth):
        arr=np.zeros(c_dims)
        arr[0]=1
        np.random.shuffle(arr)
        label=torch.cat((label,torch.tensor(arr).float().unsqueeze(0)),dim=0)
    return label
def classification_loss(logit,target):
    """
    Args:
        logits (tensor): outputs
        target (tensor): obvious
    """
    return F.binary_cross_entropy_with_logits(logit.float(),target.float(),size_average=False)/logit.float().size(0)

D_=Discriminator().to(device)
G_=Generator().to(device)
optimD=optim.Adam(D_.parameters(),lr=args.dis_lr,betas=(0.5,0.999))
optimG=optim.Adam(G_.parameters(),lr=args.gen_lr,betas=(0.5,0.999))
lambda1=lambda epoch: (-(1e-5)*epoch + 2e-4)
if num_epochs>=10:
    schedulerD=optim.lr_scheduler.LambdaLR(optimD,lambda1)
    schedulerG=optim.lr_scheduler.LambdaLR(optimG,lambda1)

g_losses=[]
d_losses=[]

warnings.filterwarnings("ignore")
for epoch in range(args.epochs):
    
    for i,data in enumerate(tqdm.notebook.tqdm(data_loader)):
        real_image, orig_labels = data[0].to(device),data[1].to(device)
        running_g_loss=.0
        running_d_loss=.0
        start=time.time()
        # Training discriminator
        optimG.zero_grad(),G_.zero_grad()
        optimD.zero_grad(),D_.zero_grad()
        target_labels=fakeLabels(orig_labels.size(0))
        D_src_real,D_cls_real=D_(real_image)

        pred=G_(real_image,target_labels)
        D_src_pred,D_cls_pred=D_(pred.detach())

        loss_adv=-torch.mean(D_src_real)+torch.mean(D_src_pred) # Adversarial Loss
        loss_cls_real=classification_loss(D_cls_pred,orig_labels) # Domain Classification Real Loss
        loss=loss_adv+lamb_cls*loss_cls_real

        D_.zero_grad(),G_.zero_grad()
        loss.backward()
        optimD.step()
        running_d_loss+=loss.item()
        # Training Generator
        if (i+1)%args.d_times==0:
            D_src_real,D_cls_real=D_(real_image)
            
            target_labels=fakeLabels(orig_labels.size(0))
            pred=G_(real_image,target_labels)
            D_src_pred,D_cls_pred=D_(pred)

            recombined=G_(pred,orig_labels)


            loss_cls_fake=classification_loss(D_cls_pred,target_labels)
            loss_adv=-torch.mean(D_src_pred)
            loss_rec=torch.mean(torch.abs(real_image-recombined))
            loss=loss_adv+lamb_cls*loss_cls_fake+lamb_rec*loss_rec
            
            D_.zero_grad(),G_.zero_grad()
            loss.backward()
            optimG.step()
            running_g_loss+=loss.item()

        if (i+1)%num_intervals==0:
            print('[%d/%d] iter:%d gen_loss:%.4f dis_loss:%.4f elapsed:%.4f'%(epoch+1,num_epochs,i+1,running_g_loss,running_d_loss,time.time()-start))

        d_losses.append(running_d_loss)
        g_losses.append(running_g_loss)
    if (epoch+1)>=10:
        schedulerD.step()
        schedulerG.step()

plotter(g_losses,d_losses)
evaluate(args.eval_idx, [0,0,1,0,1])

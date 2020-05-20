import numpy as np

import torch
import torch.utils.data

import torchvision
from torchvision import datasets
import torchvision.models
import torchvision.transforms as transforms

def get_loader(batch_size, num_workers, image_size, ndata):
    
    if ndata == 'cifar10':  
      dataset = datasets.CIFAR10(root='./content', train=True, download=True,
                               transform=transforms.Compose([transforms.Resize(image_size),
                                                             transforms.CenterCrop(image_size),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))

    if ndata == 'mnist':
      dataset = datasets.MNIST(root='./content', train=True, download=True,
                               transform=transforms.Compose([transforms.Resize(image_size),
                                                             transforms.CenterCrop(image_size),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5,), (0.5,))]))

    if ndata == 'fmnist':
      dataset = datasets.FashionMNIST(root='./content', train=True, download=True,
                               transform=transforms.Compose([transforms.Resize(32),
                                                             transforms.CenterCrop(32),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5,), (0.5,))]))


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers= num_workers ,shuffle=True)

    return dataloader


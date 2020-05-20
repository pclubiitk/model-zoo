import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dataloader(batch_size,dataroot,dataset_name,image_size):
  if dataset_name == 'mnist':
  
    transform=transforms.Compose([ transforms.Resize(image_size),transforms.CenterCrop(image_size),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
    dataset=torchvision.datasets.MNIST(root=dataroot, train=True,transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
  
  if dataset_name == 'cifar10':
  
    transform=transforms.Compose([ transforms.Resize(image_size),transforms.CenterCrop(image_size),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
    dataset=torchvision.datasets.CIFAR10(root=dataroot, train=True,transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
  

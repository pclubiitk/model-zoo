import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import time
import torchvision.transforms as transforms
import torchvision
import torchvision.transforms as transforms
import torchvision
import torch


def get_loader(batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="../../data/", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="../../data/", train=False, transform=test_transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=100, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False
    )


return train_loader, test_loader
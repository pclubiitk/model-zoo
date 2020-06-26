import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as ds


def get_data(model_path, batch_size, num_workers):

    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = ds.MNIST(model_path+'/mnist/', train='train',
                                download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

    return train_loader

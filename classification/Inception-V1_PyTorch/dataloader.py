import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets


def load_cifar():

    transform = transforms.Compose([transforms.Resize((96, 96)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    train_dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=transform)

    # Split dataset into training set and validation set.
    train_dataset, val_dataset = random_split(train_dataset, (45000, 5000))

    print("Image Shape: {}".format(
        train_dataset[0][0].numpy().shape), end='\n\n')
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)))
    print("Test Set:       {} samples".format(len(test_dataset)))

    if torch.cuda.is_available():
        BATCH_SIZE = 1024
    else:
        BATCH_SIZE = 32

    # Create iterator.
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Delete the data/ folder.
    shutil.rmtree('./data')

    return train_loader, val_loader, test_loader

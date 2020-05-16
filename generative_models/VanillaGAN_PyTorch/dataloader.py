import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

def get_loader(batch_size,num_workers):

    compose = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((.5,), (.5,))
            ])
    data = datasets.MNIST(root='./Data', train=True, transform=compose, download=True)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader


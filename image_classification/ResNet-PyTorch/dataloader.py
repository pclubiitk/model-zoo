import torchvision.transforms as transforms
import torchvision
import torch

def get_loader(batch_size,num_workers):


    transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root = './data', download = True, train = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root = './data', download = True, train = False, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers=num_workers)

    return trainloader,testloader

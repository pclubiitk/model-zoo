from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

def load_data(root_dir, batch_size, num_workers):
    """ Load the CIFAR10 dataset. """
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = datasets.CIFAR10(
        root=root_dir, 
        transform=transform, 
        download=True
        )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers)
    
    return dataloader

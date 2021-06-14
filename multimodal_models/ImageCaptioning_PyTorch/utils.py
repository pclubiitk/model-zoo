import torch
import torchvision.models
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename="drive/MyDrive/saved_checkpoint.pt"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def save_train_dataloader(state, filename="drive/MyDrive/saved_train_dataloader.pt"):
    print("=> Saving Train Dataloader")
    torch.save(state, filename, map_location = device)

def save_test_dataloader(state, filename="drive/MyDrive/saved_test_dataloader.pt"):
    print("=> Saving Test Dataloader")
    torch.save(state, filename, map_location = device)

def save_train_dataset(state, filename="drive/MyDrive/saved_train_dataset.pt"):
    print("=> Saving Train Dataset")
    torch.save(state, filename, map_location = device)

def save_test_dataset(state, filename="drive/MyDrive/saved_test_dataset.pt"):
    print("=> Saving Test Dataset")
    torch.save(state, filename, map_location = device)
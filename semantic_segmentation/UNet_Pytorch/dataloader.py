import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from skimage import io, transform


class Brain_data(Dataset):
    def __init__(self, path):
        self.path = path
        self.patients = [
            file for file in os.listdir(path) if file not in ["data.csv", "README.md"]
        ]
        self.masks, self.images = [], []

        for patient in self.patients:
            for file in os.listdir(os.path.join(self.path, patient)):
                if "mask" in file.split(".")[0].split("_"):
                    self.masks.append(os.path.join(self.path, patient, file))
                else:
                    self.images.append(os.path.join(self.path, patient, file))

        self.images = sorted(self.images)
        self.masks = sorted(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = io.imread(image)
        image = transform.resize(image, (256, 256))
        image = image / 255
        image = image.transpose((2, 0, 1))

        mask = io.imread(mask)
        mask = transform.resize(mask, (256, 256))
        mask = mask / 255
        mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return (image, mask)


def get_loader(batch_size, num_workers):
    data_folder = "/content/kaggle_3m"

    data = Brain_data(data_folder)
    train_set, val_set = random_split(data, [3200, 729])

    train_dl = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dl = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    return train_dl, val_dl

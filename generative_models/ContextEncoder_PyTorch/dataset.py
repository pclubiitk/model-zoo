import glob
import os
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms



class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # top left coordinate in image
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            #use random crop in case of train
            masked_img, aux = self.apply_random_mask(img)
        else:
           #crop center in case of test data
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)

def get_loader(train_batch_size,test_batch_size,mask_size,dataset,num_workers):
  transforms_ = [
    transforms.Resize((128,128), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ]
  PATH = "./data/"  + dataset
  dataloader = DataLoader(
    ImageDataset(PATH, transforms_=transforms_,mask_size=mask_size),
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
  )
  test_dataloader = DataLoader(
    ImageDataset(PATH, transforms_=transforms_, mode="val"),
    batch_size=test_batch_size,
    shuffle=True,
    num_workers=num_workers,
  )        
  return dataloader,test_dataloader
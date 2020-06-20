from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

def clean_dataset(dir):
    """ Remove images which are not in RGB colour space"""
    for img in os.listdir(dir):
        path = os.path.join(dir, img)
        im = Image.open(path)
        if(im.mode != 'RGB'):
            os.remove(path)

class TrainDataset(Dataset):

    def __init__(self, dir):
        super().__init__()
        clean_dataset(dir)
        self.img = [os.path.join(dir, x) for x in os.listdir(dir)]
        self.hr = transforms.Compose([
                                    transforms.RandomCrop(96, pad_if_needed=True),
                                    transforms.ToTensor(),
        ])
        self.lr = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(24, interpolation=Image.BICUBIC),
                                    transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        hr_image = self.hr(Image.open(self.img[index]))
        lr_image = self.lr(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.img)

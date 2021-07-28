import torch
from fiftyone.zoo import load_zoo_dataset
from PIL import Image

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


class FiftyOneDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset."""

    def __init__(self, fiftyone_dataset, img_size=256, upscale_factor=1):
        self.samples = fiftyone_dataset
        self.img_size = img_size
        self.upscale_factor = upscale_factor
        self.img_paths = self.samples.values("filepath")

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        target = self.transform(img.copy(), self.img_size)
        img = self.transform(img, self.img_size, self.upscale_factor)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def transform(self, img, img_size, upscale_factor=1):
        img_transform = transforms.Compose(
            [
                transforms.Resize(img_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size // upscale_factor),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )
        return img_transform(img)


def load_dataset(batch_size, img_size, num_workers, split="test", upscale_factor=1):
    """Load the coco-2017 dataset."""
    dataset = load_zoo_dataset("coco-2017", split=split)
    dataset = FiftyOneDataset(dataset, img_size, upscale_factor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True
    )
    return dataloader

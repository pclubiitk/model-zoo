import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from helper_functions.losses import custom_loss
from torch.autograd import Variable
import helper_functions.config as cfg

def downscale_16times(ndf = cfg.discriminatorDim, inn_channels = cfg.channels):
    encode_img = nn.Sequential(
        nn.Conv2d(inn_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img

class condAugmentation(nn.Module):
    def __init__(self):
        super(condAugmentation, self).__init__()
        self.CD = cfg.embeddingsDim # 1024, 512 : 256, 512 / 4
        self.fc = nn.Linear(cfg.textDim, self.CD * 4, bias=True)
        self.relu = custom_loss()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.CD]
        logvar = x[:, self.CD:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

def ret_image(img_path, imsize, StageNum, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(StageNum):
        if i < 2:
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, size=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.size = size

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, size=self.size)
        return x
import torch
import torch.nn as nn
from helper_functions.losses import custom_loss
from helper_functions.ret_image import Interpolate

def downBlock(inn, out):
    block = nn.Sequential(
        nn.Conv2d(inn, out, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def upScale(inn, out):
    block = nn.Sequential(
        Interpolate(scale_factor=2, mode="nearest"),
        nn.Conv2d(inn, out * 2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out * 2),
        custom_loss()
    )
    return block

def Block3x3_leakRelu(inn, out):
    block = nn.Sequential(
        nn.Conv2d(inn, out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def normalBlock(inn, out):
    block = nn.Sequential(
        nn.Conv2d(inn, out * 2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out * 2),
        custom_loss()
    )
    return block

class Residual(nn.Module):
    def __init__(self, channel):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, channel * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel * 2),
            custom_loss(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel))

    def forward(self, x):
        y = x.clone()
        return self.block(x) + y
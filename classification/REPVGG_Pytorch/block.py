import torch
import torch.nn as nn
import os
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random
import time
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import time
import torchvision.transforms as transforms
import torchvision


def fcbn(conv: nn.Conv2d, bn: nn.BatchNorm2d):

    sf = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
    fb = bn.bias.data - sf * bn.running_mean
    fb = fb + sf * conv.bias.data
    fk = sf.view(-1, 1, 1, 1) * conv.weight.data

    return fk, fb


class block(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, stride=None):
        super(block, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nl = nn.ReLU()
        # self.s = nn.Identity()
        b1 = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(3, 3),
                stride=self.stride,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(num_features=self.out_channels),
        ]
        self.x = nn.Sequential(*b1)
        b2 = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                stride=self.stride,
                padding=(0, 0),
            ),
            nn.BatchNorm2d(num_features=self.out_channels),
        ]
        self.y = nn.Sequential(*b2)
        self.branches = nn.ModuleList([nn.Sequential(*b1), nn.Sequential(*b2)])
        if self.stride == (1, 1):
            self.z = nn.BatchNorm2d(num_features=self.in_channels)
            self.branches.append(nn.BatchNorm2d(num_features=self.in_channels))
        else:
            self.z = None

    def reparam(self):
        in_ch = self.branches[0][0].weight.data.shape[1]
        out_ch = self.branches[0][0].weight.data.shape[0]

        r = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=True,
            stride=self.branches[0][0].stride,
        )

        fk3, fb3 = fcbn(*self.branches[0])
        fk1, fb1 = fcbn(*self.branches[1])
        rwd = fk3
        rbd = fb3
        rwd[..., 1:2, 1:2] = rwd[..., 1:2, 1:2] + fk1

        # in case we apply bn layer
        if len(self.branches) == 3:
            sf = (
                self.branches[2].weight.data
                / (self.branches[2].running_var + self.branches[2].eps).sqrt()
            )
            rwd[range(out_ch), range(in_ch), 1, 1] += sf
            rbd = rbd + self.branches[2].bias.data
            rbd = rbd - sf * self.branches[2].running_mean

        self.branches = nn.ModuleList([r])

    def forward(self, input):
        # case of reparametrization
        if len(self.branches) == 1:
            out = self.branches[0](input)
        else:
            if self.z == None:
                out = self.x(input) + self.y(input)
            else:
                out = self.x(input) + self.y(input) + self.z(input)
                out = self.nl(out)
        return out

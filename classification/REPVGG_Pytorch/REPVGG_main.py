from block import fcbn
from block import block
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


class REPVGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, blocks=None, multipl=None):
        super(REPVGG, self).__init__()
        self.blocks = blocks
        self.multipl = multipl
        self.in_channels = in_channels
        # self.main_REPVGG = self.main_architecture()
        self.g = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * multipl[3]), num_classes)
        layers = []

        # stage0

        multipl = self.multipl
        out_channels = min(64, int(64 * multipl[0]))
        in_channels = self.in_channels
        blocks = self.blocks
        layers += [
            block(in_channels=in_channels, out_channels=out_channels, stride=(2, 2))
        ]
        in_channels = min(64, int(64 * multipl[0]))

        # stage 1,2 ,3 4

        for i in range(4):

            out_channels = int(64 * (2 ** i) * multipl[i])
            layers += [
                block(in_channels=in_channels, out_channels=out_channels, stride=(2, 2))
            ]
            in_channels = out_channels

            for j in range(blocks[i] - 1):
                layers += [
                    block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(1, 1),
                    )
                ]
            in_channels = out_channels
        self.layers = layers
        self.main_REPVGG = nn.Sequential(*layers)

    def reparametrize(self):
        for block in self.layers:
            block.reparam()

    def forward(self, x):
        x = self.main_REPVGG(x)
        x = self.g(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

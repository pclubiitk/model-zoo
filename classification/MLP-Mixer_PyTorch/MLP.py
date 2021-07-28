from MLP_MIXER_Block import MixerBlock
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


class MLPMixer(nn.Module):
    def __init__(self,input_size, patch_size, dim = 512, img_channel=3, layers = 12, num_classes=12):
        super().__init__()
        patch = int(input_size[0]/patch_size[0] * input_size[1]/patch_size[1])
        patch_dim = img_channel * patch_size[0] * patch_size[1]
        self.embedding = nn.Sequential(
                                                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),
                                                nn.Linear(patch_dim, dim)
                                                )
        self.main_architecture = nn.Sequential(*[nn.Sequential(MixerBlock(dim,patch)) for _ in range(layers)])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(dim,num_classes)
    
    def forward(self,x):
        x = self.embedding(x)
        x = self.main_architecture(x)
        x = self.pool(x.transpose(1,2))
        x = self.classifier(x).squeeze(2))
        return x;
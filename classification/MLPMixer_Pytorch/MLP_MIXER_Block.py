import torch
import torch.nn as nn
import os
import time
import importlib
from einops.layers.torch import Rearrange
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


class MixerBlock(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.pre_layer_norm = nn.LayerNorm(dim)
        self.post_layer_norm = nn.LayerNorm(dim)
        
        self.token_mixer = nn.Sequential(
                            nn.Linear(patch, dim),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(dim, patch),
                            nn.Dropout(0.1)
                            )
        
        self.channel_mixer = nn.Sequential(
                            nn.Linear(dim, dim),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(dim, dim),
                            nn.Dropout(0.1)
                            )
    def forward(self, x):
        z =self.pre_layer_norm(x)
        y = self.token_mixer(z.transpose(1,2)).transpose(1,2)
        y = y + x
        ln = self.post_layer_norm(y)
        out = self.channel_mixer(ln)+y
        return out
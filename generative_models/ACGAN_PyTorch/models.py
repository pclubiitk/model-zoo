import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator Network
    Inputs : noise, class label
    Output : (-1, 3, 32, 32) Tensor
    """
    def __init__(self, nz):
        super().__init__()
        self.nz = nz
        self.fc1 = nn.Linear(nz, 384)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(-1, self.nz)
        x = self.fc1(x)
        x = x.view(-1, 384, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator Network
    Inputs : Image Tensor
    Output : src - classifies input as real / fake
             cls - predicted class of the input image
    """
    def __init__(self):
        super().__init__()
        classes = 10
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        self.fc_source = nn.Sequential(
            nn.Linear(4*4*512, 1),
            nn.Sigmoid()
        )
        self.fc_class = nn.Sequential(
            nn.Linear(4*4*512, classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 4*4*512)
        src = self.fc_source(x).view(-1,1).squeeze(1)
        cls = self.fc_class(x)
        return src, cls

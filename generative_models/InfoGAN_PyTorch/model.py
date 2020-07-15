import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Architecture as mentioned in InfoGAN paper for MNIST dataset.

    Link: https://arxiv.org/pdf/1606.03657.pdf
"""


class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(74, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 7*7*128)
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 128, 7, 7)
        x = self.layer3(x)
        # nn.functional.tanh is deprecated. Use torch.tanh instead.
        x = torch.tanh(self.layer4(x))

        return x


class SharedNetwork(nn.Module):

    def __init__(self):

        super(SharedNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 128*7*7)
        x = self.layer3(x)
        return x


class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 1)
        )

    def forward(self, x):

        # nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
        x = torch.sigmoid(self.layer4(x))
        return x


class Recogniser(nn.Module):

    def __init__(self):

        super(Recogniser, self).__init__()
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1)
        )
        self.layer5_disc = nn.Sequential(
            nn.Linear(128, 10)
        )
        self.layer5_mu = nn.Sequential(
            nn.Linear(128, 2)
        )
        self.layer5_var = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, x):

        x = self.layer4(x)
        disc_logits = self.layer5_disc(x)
        mu = self.layer5_mu(x)
        var = self.layer5_var(x)
        return disc_logits, mu, var


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)

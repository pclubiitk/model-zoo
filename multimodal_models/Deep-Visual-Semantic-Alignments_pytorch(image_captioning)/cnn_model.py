import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

class Resnet18(nn.Module):
    def __init__(self, embedding_dim):
        super(Resnet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        modules = list(self.resnet18.children())[:-1] #leaving last softmax
        self.resnet18 = nn.Sequential(*modules)
        self.linear = nn.Linear(in_features, embedding_dim)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0,0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        embed = self.resnet18(images)
        embed = Variable(embed)
        embed = self.linear(embed.view(embed.size(0), -1))
        return embed

class Inception(nn.Module):
    def __init__(self, embedding_dim):
        super(Inception, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        in_features = self.inception.fc.in_features
        self.linear = nn.Linear(in_features, embedding_dim)
        self.inception.fc = self.linear
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        embed = self.inception(images)
        return embed


def get_CNN(architecture, embedding_dim):
    if architecture == 'resnet18':
        cnn = Resnet18(embedding_dim = embedding_dim)
    elif architecture == 'inception':
        cnn = Inception(embedding_dim = embedding_dim)

    return cnn

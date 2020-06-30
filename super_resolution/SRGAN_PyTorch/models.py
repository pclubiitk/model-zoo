import torch.nn as nn

class FeatureExtractor(nn.Module):
    """Extracts feature vectors from VGG-19"""  
    def __init__(self, model, i, j):
        super().__init__()
        maxpool = [4, 9, 18, 27, 36]
        layer = maxpool[i-1]-2*j
        self.features = nn.Sequential(*list(model.features.children())[:(layer+1)])
        
    def forward(self, x):
        return self.features(x)

class ResidualBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return x + y

class UpsampleBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 256, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.relu(x)
        return x

class Generator(nn.Module):

    def __init__(self, b):
        super().__init__()
        self.b = b
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.relu = nn.ReLU()
        for i in range(b):
            self.add_module(f'ResidualBlock_{i+1}', ResidualBlock())
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn = nn.BatchNorm2d(64)
        for i in range(2):
            self.add_module(f'UpsampleBlock_{i+1}', UpsampleBlock())
        self.conv3 = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        y = x.clone()
        for i in range(self.b):
            y = self.__getattr__(f'ResidualBlock_{i+1}')(y)
        y = self.conv2(y)
        y = self.bn(y)
        y = y + x
        for i in range(2):
            y = self.__getattr__(f'UpsampleBlock_{i+1}')(y)
        y = self.conv3(y)
        return y

class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = self.bn(x)
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.add_module('DiscriminatorBlock1', DiscriminatorBlock(64, 64, 2))
        n = 128
        for i in range(3):
            self.add_module(f'DiscriminatorBlock{2+2*i}', DiscriminatorBlock(n//2, n, 1))
            self.add_module(f'DiscriminatorBlock{3+2*i}', DiscriminatorBlock(n, n, 2))
            n *= 2
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        for i in range(7):
            x = self.__getattr__(f'DiscriminatorBlock{i+1}')(x)
        x = x.view(-1,512*6*6)
        x = self.leaky_relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = x.view(-1,1)
        return x

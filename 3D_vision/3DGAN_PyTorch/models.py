import torch
import torch.nn as nn

class generator(nn.Module):
    """
    Generator for 3D-GAN-
    The generator consists of five fully convolution layers with numbers of channels
    {512, 256, 128, 64, 1}, kernel sizes {4, 4, 4, 4, 4}, and strides {1, 2, 2, 2, 2}. 
    We add ReLU and batch normalization layers between convolutional layers, and a Sigmoid
    layer at the end. The input is a 200-dimensional vector, and the output is a 64 × 64 × 64 
    matrix with values in [0, 1].
    """
    def __init__(self):
        super().__init__()
        self.layer1=nn.Sequential(
            nn.ConvTranspose3d(in_channels=200,out_channels=512,kernel_size=2,stride=1,bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
        )
        self.layer2=nn.Sequential(
            nn.ConvTranspose3d(in_channels=512,out_channels=256,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
        )
        self.layer3=nn.Sequential(
            nn.ConvTranspose3d(in_channels=256,out_channels=128,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
        )
        self.layer4=nn.Sequential(
            nn.ConvTranspose3d(in_channels=128,out_channels=64,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
        )
        self.layer5=nn.Sequential(
            nn.ConvTranspose3d(in_channels=64,out_channels=1,kernel_size=2,stride=2,bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self,x):
        bsize=x.size(0)
        x=x.view(bsize,vectorSize,1,1,1)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        return x


class discriminator(nn.Module):
    """
    Discriminator for 3D-GAN-
    As a mirrored version of the generator, the discriminator takes as input a 
    64 × 64 × 64 matrix, and outputs a real number in [0, 1]. The discriminator 
    consists of 5 volumetric convolution layers, with numbers of channels {64,128,256,512,1}, 
    kernel sizes {4,4,4,4,4}, and strides {2, 2, 2, 2, 1}. There are leaky ReLU layers of 
    parameter 0.2 and batch normalization layers in between, and a Sigmoid layer at the end.
    """
    def __init__(self):
        super().__init__()
        self.layer1=nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=64,kernel_size=3,stride=2,bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.layer2=nn.Sequential(
            nn.Conv3d(in_channels=64,out_channels=128,kernel_size=3,stride=2,bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.layer3=nn.Sequential(
            nn.Conv3d(in_channels=128,out_channels=256,kernel_size=3,stride=2,bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.layer4=nn.Sequential(
            nn.Conv3d(in_channels=256,out_channels=512,kernel_size=3,stride=2,bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.layer5=nn.Sequential(
            nn.Conv3d(in_channels=512,out_channels=1,kernel_size=1,stride=1,bias=False),
            nn.Sigmoid()
        )
 
    def forward(self,x):
        bsize=x.size(0)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=x.view(bsize,1)
        return x


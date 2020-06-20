import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator module:

    Especially for Celeb-A dataset, otherwise cls vector size will vary.

    Args:
        Image Tensor (-1,3,128,128)

    Output:
        src (float): Probability between 0 and 1, discriminates whether source is real or fake.
        cls (tensor, shape(c_dims,)): Returns class-wise probability, similar to that of AC-GAN (Goodfellow et.al).
    """

    def __init__(self):
        super().__init__()
        self.input_layer=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU()
        )
        self.hidden1=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU()
        )
        self.hidden2=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU()
        )
        self.hidden3=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU()
        )
        self.hidden4=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU()
        )
        self.hidden5=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=2048,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU()
        )
        self.src=nn.Sequential(
            nn.Conv2d(in_channels=2048,out_channels=1,kernel_size=(3,3),stride=1,padding=1,bias=False)
        )
        self.cls=nn.Sequential(
            nn.Conv2d(in_channels=2048,out_channels=c_dims,kernel_size=(1,1),stride=1,padding=0,bias=False)
        )
    def forward(self,x):
        bsize=x.size(0)
        x=self.input_layer(x)
        x=self.hidden1(x)
        x=self.hidden2(x)
        x=self.hidden3(x)
        x=self.hidden4(x)
        x=self.hidden5(x)
        src=self.src(x)
        cls=self.cls(x).squeeze()
        return src,cls

class Generator(nn.Module):
    """
    Generator module:

    Args:
        x : Image Tensor (-1,3,128,128)
        c : Label Tensor (-1,c_dims)

    Output:
        Image Tensor (-1,3,128,128)
    """
    def __init__(self):
        super().__init__()
        self.down_sample=nn.Sequential(
            nn.Conv2d(in_channels=3+c_dims,out_channels=64,kernel_size=(7,7),stride=1,padding=3,bias=False),
            nn.InstanceNorm2d(64,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(128,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(4,4),stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(256,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        bottle_neck=[]
        for _ in range(2):
            bottle_neck.append(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=1,padding=1,bias=False))
            bottle_neck.append(nn.InstanceNorm2d(256,affine=True,track_running_stats=True))
            bottle_neck.append(nn.ReLU(inplace=True))
        self.bottleneck=nn.Sequential(*bottle_neck)
        self.up_sample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(4,4),stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(128,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(64,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(7,7),stride=1,padding=3,bias=False),
            nn.Tanh(),
        )

    def forward(self,x,c):
        c=c.view(c.size(0),c.size(1),1,1).float()
        c=c.repeat(1,1,x.size(2),x.size(3))
        x=torch.cat((x,c),dim=1)
        x=self.down_sample(x)
        x=self.bottleneck(x)
        x=self.up_sample(x)
        return x

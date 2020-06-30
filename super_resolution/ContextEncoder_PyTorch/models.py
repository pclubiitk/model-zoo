import torch
import torch.nn as nn


#Generator model
class Generator(nn.Module):

  def __init__(self,nc):

    super(Generator,self).__init__()

    def Downsample(in_channels,out_channels,normalize = True):

      layer = [nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1)]
      if normalize :
        layer.append(nn.BatchNorm2d(out_channels))
      layer.append(nn.LeakyReLU(0.2,inplace=True))

      return layer

    def Upsample(in_channels,out_channels,normalize = True):

      layer = [nn.ConvTranspose2d(in_channels,out_channels,kernel_size= 4,stride=2,padding=1)]
      if normalize : 
        layer.append(nn.BatchNorm2d(out_channels))
      layer.append(nn.LeakyReLU(0.2,True))

      return layer

    self.main = nn.Sequential(
        *Downsample(nc,64,normalize=False),
        *Downsample(64,64),
        *Downsample(64,128),
        *Downsample(128,256),
        *Downsample(256,512),
        nn.Conv2d(512,100,4),  # Bottleneck
        nn.ConvTranspose2d(100,512,4),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2,True),
        *Upsample(512,256),
        *Upsample(256,128),
        *Upsample(128,64),
        *Upsample(64,nc,normalize=False),
        nn.Tanh()
    )    

  def forward(self,x):

      return self.main(x)

#Discriminator model 
class Discriminator(nn.Module):
  
    def __init__(self,nc):

        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        
        layers.extend(discriminator_block(nc,64,False))
        layers.extend(discriminator_block(64,128,True))
        layers.extend(discriminator_block(128,256,True))
        layers.extend(discriminator_block(256,512,True))
        layers.append(nn.Conv2d(512,1,4))
        self.model = nn.Sequential(*layers)                          
        

    def forward(self, img):
        x = self.model(img)
        return x.view(x.size(0),-1)      
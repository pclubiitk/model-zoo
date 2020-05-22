import torch
import torch.nn as nn

# weight initialisation with mean=0 and stddev=0.02

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            
            nn.ConvTranspose2d(channels_noise, features_g*8, kernel_size=4, stride=1, padding=0 , bias = False ),
            nn.BatchNorm2d(features_g*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1 , bias = False ),
            nn.BatchNorm2d(features_g*4),
              nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1  , bias = False),
            nn.BatchNorm2d(features_g*2),
             nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g*2, features_g, kernel_size=4, stride=2, padding=1  , bias = False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1 , bias = False ),
            nn.Tanh()
            )      
    
    def forward(self, x):
        return self.net(x)        
        
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1 , bias = False ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d, features_d*2, kernel_size=4, stride=2, padding=1 , bias = False ),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1 , bias = False ),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1 , bias = False ),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0 , bias = False ),
            nn.Sigmoid()
            )  
    
    def forward(self, x):
        return self.net(x)
        
        

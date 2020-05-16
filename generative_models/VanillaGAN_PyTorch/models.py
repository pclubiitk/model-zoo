import torch
import torch.nn as nn

class Generator(nn.Module):

  """
  the generator in a GAN
  """
  def __init__(self,n_features=100):
    super().__init__()

    n_out = 784
    self.hidden0 = nn.Sequential(
        nn.Linear(n_features,256),
        nn.LeakyReLU(0.2)
    )
    self.hidden1 = nn.Sequential(
        nn.Linear(256,512),
        nn.LeakyReLU(0.2)
    )
    self.hidden2 = nn.Sequential(
        nn.Linear(512,1024),
        nn.LeakyReLU(0.2)
    )
    self.out = nn.Sequential(
        nn.Linear(1024,n_out),
        nn.Tanh()
    )

  def forward(self,x):

    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    return x

class Discriminator(nn.Module):

  def __init__(self):

    super().__init__()
    n_features = 784
    n_out = 1

    self.hidden0 = nn.Sequential(
        nn.Linear(n_features,1024),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
    )
    self.hidden1 = nn.Sequential(
        nn.Linear(1024,512),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
    )
    self.hidden2 = nn.Sequential(
        nn.Linear(512,256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
    )
    self.out = nn.Sequential(
        nn.Linear(256,n_out),
        nn.Sigmoid()
    )    
  def forward(self, x):
    x = x.view(-1,784)
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    return x


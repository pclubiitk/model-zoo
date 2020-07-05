import torch.nn as nn
import torch
import torch.nn.functional as F

# We will use a normal deterministic encoder, which is same as the one used in an ordinary autoencoder
class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Sequential(
            nn.Linear(784,1000),
            nn.Dropout(p=.25),
            nn.ReLU(True),
            nn.Linear(1000,1000),
            nn.Dropout(p=.25),
            nn.ReLU(True),
            nn.Linear(1000,8),
        )

    def forward(self,x):
        bsize=x.size(0)
        x=x.view(bsize,-1)
        return self.block(x)

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Sequential(
            nn.Linear(8,1000),
            nn.Dropout(p=.25),
            nn.ReLU(True),
            nn.Linear(1000,1000),
            nn.Dropout(p=.25),
            nn.ReLU(True),
            nn.Linear(1000,784),
        )
    
    def forward(self,x):
        x=self.block(x)
        return F.sigmoid(x)

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Sequential(
            nn.Linear(8,1000),
            nn.Dropout(p=.2),
            nn.ReLU(True),
            nn.Linear(1000,1000),
            nn.Dropout(p=.2),
            nn.ReLU(True),
            nn.Linear(1000,1)
        )
    def forward(self,x):
        x=self.block(x)
        return F.sigmoid(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self,args):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(10,10)
        self.layer = 128
        self.channel = args.channel
        self.latent_dim = args.latent_dim
        self.image_size = args.image_size

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim+args.num_class, self.layer),
            nn.BatchNorm1d(self.layer,0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.layer, self.layer*2),
            nn.BatchNorm1d(self.layer*2,0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.layer*2, self.layer*4),
            nn.BatchNorm1d(self.layer*4,0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.layer*4, self.layer*8),
            nn.BatchNorm1d(self.layer*8,0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.layer*8,args.channel*args.image_size*args.image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        # print(noise.shape)
        z = noise.view(noise.size(0), self.latent_dim)
        x = torch.cat([c, z], 1)
        out = self.model(x)
        return out.view(out.size(0), self.channel, self.image_size, self.image_size)

class Discriminator(nn.Module):
      def __init__(self,args):
          super(Discriminator, self).__init__()
          self.label_embedding = nn.Embedding(args.num_class, args.num_class)
          self.layer = 256

          self.model = nn.Sequential(
              nn.Linear(args.num_class + (args.channel * args.image_size * args.image_size), self.layer * 4),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Dropout(0.4),
              nn.Linear(self.layer * 4, self.layer * 2),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Dropout(0.4),
              nn.Linear(self.layer * 2, self.layer),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Dropout(0.4),
              nn.Linear(self.layer, 1),
              nn.Sigmoid()
          )

      def forward(self, img, label):
          x = img.view(img.size(0), -1)
          # print(x.shape)
          z = self.label_embedding(label)
          x = torch.cat([x, z], 1)
          out = self.model(x)
          return out

def init_weights(m):
    if type(m)==nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
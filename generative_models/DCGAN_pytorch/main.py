import torch
import torchvision
import torch.nn as nn                       # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim                 # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.datasets as datasets     # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perform on our dataset
from torch.utils.data import DataLoader     # Gives easier dataset managment and creates mini batches

from models import Generator,Discriminator,weights_init
from utils import compare_img,plot_loss,animation

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
image_size = args.image_size
channels_noise = args.channels_noise
num_epochs = args.num_epochs

features_d = 128
features_g = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    ])

dataset = datasets.MNIST(root='dataset/', train=True, transform=my_transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)
netG=netG.apply(weights_init)
netD=netD.apply(weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999) )
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999) )

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

img_list = []
G_losses = []
D_losses = []

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        label = (torch.ones(batch_size)*0.9).to(device)
        output = netD(data).view(-1)
        lossD_real = criterion(output, label)
        D_x = output.mean().item()
        
        noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
        fake = netG(noise)
        label = (torch.ones(batch_size)*0.1).to(device)
        output = netD(fake.detach()).view(-1)
        lossD_fake = criterion(output, label)
        
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
        
        ### Train Generator: max log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()
        D_G_x = output.mean().item()
        
         
        if batch_idx % 100 == 0:
            # Print losses ocassionally
            print(f'Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)}  Loss D: {lossD:.4f}  loss G: {lossG:.4f}  D(x): {D_x:.4f}  D(G(z)): {D_G_x:.4f} ')
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))


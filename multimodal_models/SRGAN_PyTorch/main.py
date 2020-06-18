import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, utils
from torch.utils.data import DataLoader
import time
from dataloader import TrainDataset
from models import FeatureExtractor, Generator, Discriminator
from torchsummary import summary
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./', help='path to dataset')
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--pre_num_epochs', type=int, default=100, help='number of pre-training epochs')
parser.add_argument('--outdir', default='./', help='directory to output model checkpoints')
parser.add_argument('--load_checkpoint', default=0, type=int, help='Pass 1 to load checkpoint')
parser.add_argument('--b', default=16, type=int, help='number of residual blocks in generator')
args = parser.parse_args()

# Load data
dataset = TrainDataset(args.root_dir)
dataloader = DataLoader(dataset, args.batch_size, True, num_workers=args.num_workers)
# Initialize models
vgg = models.vgg19(pretrained=True)
feature_extractor = FeatureExtractor(vgg, 5, 4)
if torch.cuda.device_count() > 1:
    feature_extractor = nn.DataParallel(feature_extractor)
feature_extractor = feature_extractor.to(device)

disc = Discriminator()
if torch.cuda.device_count() > 1:
    disc = nn.DataParallel(disc)
disc = disc.to(device)
if args.load_checkpoint == 1 and os.path.exists('disc.pt'):
    disc.load_state_dict(torch.load('disc.pt'))
print(disc)

gen = Generator(args.b)
if torch.cuda.device_count() > 1:
    gen = nn.DataParallel(gen)
gen = gen.to(device)
if args.load_checkpoint == 1 and os.path.exists('gen.pt'):
    gen.load_state_dict(torch.load('gen.pt'))
print(gen)

content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()
optimG = optim.Adam(gen.parameters(), args.lr)
schedulerG1 = optim.lr_scheduler.MultiStepLR(optimG, [100], 0.1)
schedulerG2 = optim.lr_scheduler.MultiStepLR(optimG, [100], 0.1)
optimD = optim.Adam(disc.parameters(), args.lr)
schedulerD = optim.lr_scheduler.MultiStepLR(optimD, [100], 0.1)
writer = SummaryWriter()
# Generator pre-training
start_time = time.time()
iters = 0
for epoch in range(args.pre_num_epochs):
    
    for i, data in enumerate(dataloader, 0):

        lr, hr_real = data
        hr_real = hr_real.to(device)
        lr = lr.to(device)

        batch_size = hr_real.size()[0]
        hr_fake = gen(lr)

        gen.zero_grad()
        gen_content_loss = content_criterion(hr_fake, hr_real)
        gen_content_loss.backward()
        optimG.step()

        if i == 0:
            print(f'[{epoch}/{args.pre_num_epochs}][{i}/{len(dataloader)}] Gen_MSE: {gen_content_loss.item()}')
        iters += 1 

    torch.save(gen.state_dict(), f'{args.outdir}gen.pt')
    schedulerG1.step()
    print(f'Time Elapsed: {(time.time()-start_time): .2f}')

# Adversarial Training
G_losses = []
D_losses = []
iters = 0
optimG = optim.Adam(gen.parameters(), args.lr)
for epoch in range(args.num_epochs):
    
    for i, data in enumerate(dataloader):
        iters += 1
        lr, hr_real = data
        batch_size = hr_real.size()[0]
        hr_real = hr_real.to(device)
        lr = lr.to(device)
        hr_fake = gen(lr)

        # Label Smoothing (Salimans et. al. 2016)
        target_real = torch.rand(batch_size, 1, device=device)*0.85+0.3
        target_fake = torch.rand(batch_size, 1, device=device)*0.15

        # Discriminator
        disc.zero_grad()
        D_x = disc(hr_real)
        D_G_z1 = disc(hr_fake.detach())
        errD_real = adversarial_criterion(D_x, target_real)
        errD_fake = adversarial_criterion(D_G_z1, target_fake)
        errD = errD_real + errD_fake
        D_x = D_x.view(-1).mean().item()
        D_G_z1 = D_G_z1.view(-1).mean().item()
        errD.backward()
        optimD.step()

        # Generator
        gen.zero_grad()
        real_features = feature_extractor(hr_real)
        fake_features = feature_extractor(hr_fake)
        ones = torch.ones(batch_size, 1, device=device)

        errG_mse = content_criterion(hr_fake, hr_real)
        errG_vgg = content_criterion(fake_features, real_features)
        D_G_z2 = disc(hr_fake)
        errG_adv = adversarial_criterion(D_G_z2, ones)
        errG = errG_mse + 0.006*errG_vgg + 0.001*errG_adv
        D_G_z2 = D_G_z2.view(-1).mean().item()
        errG.backward()
        optimG.step()
        if i == 0:
            print(f'[{epoch}/{args.num_epochs}][{i}/{len(dataloader)}] errD: {errD.item():.4f}'
                    f' errG: {errG.item():.4f} ({errG_mse.item():.4f}/{0.006*errG_vgg.item():.4f}/{0.001*errG_adv.item():.4f})'
            f' D(HR): {D_x :.4f} D(G(LR1)): {D_G_z1:.4f} D(G(LR2)): {D_G_z2:.4f}')
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    torch.save(gen.state_dict(), f'{args.outdir}gen.pt')
    torch.save(disc.state_dict(), f'{args.outdir}disc.pt')
    print(f'Time Elapsed: {(time.time()-start_time): .2f}')
    schedulerD.step()
    schedulerG2.step()

print(f'Finished Training {args.num_epochs} epochs')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

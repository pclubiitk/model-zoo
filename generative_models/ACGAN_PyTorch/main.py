import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
import matplotlib.pyplot as plt
import time
from torchsummary import summary
from dataloader import load_data
from models import Generator, Discriminator
from utils import weights_init, compute_acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./', help='path to dataset')
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--outdir', default='./', help='folder to output images and model checkpoints')

args = parser.parse_args()

outdir = args.outdir
if not os.path.exists(outdir+'images/'):
    os.makedirs(outdir+'images/')

# Load data
dataloader = load_data(args.root_dir, args.batch_size, args.num_workers)

# Initialize model
gen = Generator(args.nz).to(device)
gen.apply(weights_init)
summary(gen, (args.nz,))
optimG = optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

disc = Discriminator().to(device)
disc.apply(weights_init)
summary(disc, (3, 32, 32))
optimD = optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

criterion_src = nn.BCELoss()
criterion_cls = nn.CrossEntropyLoss()
real_label = 1.0
fake_label = 0.0
test_noise = torch.randn(64, 110, 1, 1, device=device)

def train_disc(
    disc, gen, data, device, batch_size, real_label, nz,
    fake_label, criterion_src, criterion_cls, optimD
    ):

    # Real Images
    disc.zero_grad()
    real_image, real_class = data
    real_image = real_image.to(device)
    real_class = real_class.to(device)
    src_label = torch.full((batch_size,), real_label, device=device)
    cls_label = real_class.view(batch_size,)

    src, cls = disc(real_image)
    errD_src_real = criterion_src(src, src_label)
    errD_cls_real = criterion_cls(cls, cls_label)
    errD_real = errD_src_real + errD_cls_real
    errD_real.backward()
    D_x = src.mean().item()
    accuracy = compute_acc(cls, cls_label)

    # Fake Images
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    cls_label = torch.randint(0, 2, (batch_size,), device=device)
    src_label.fill_(fake_label)

    fake = gen(noise)
    src, cls = disc(fake.detach())
    errD_src_fake = criterion_src(src, src_label)
    errD_cls_fake = criterion_cls(cls, cls_label)
    errD_fake = errD_src_fake + errD_cls_fake
    errD_fake.backward()
    D_G_z1 = src.mean().item()
    errD = errD_real + errD_fake

    optimD.step()

    return accuracy, errD.item(), D_x, D_G_z1, fake

def train_gen(
    disc, gen, real_label, fake, batch_size,
    optimG, criterion_src, criterion_cls
    ):

    gen.zero_grad()
    src_label = torch.full((batch_size,), real_label, device=device)
    cls_label = torch.randint(0, 2, (batch_size,), device=device)
    src, cls = disc(fake)
    errG_src = criterion_src(src, src_label)
    errG_cls = criterion_cls(cls, cls_label)
    errG = errG_src + errG_cls
    errG.backward()
    D_G_z2 = src.mean().item()

    optimG.step()

    return errG.item(), D_G_z2

def train(
    num_epochs, dataloader, disc, gen, device, 
    real_label, fake_label, criterion_src, criterion_cls, 
    optimD, optimG, test_noise, outdir, nz
    ):

    print('Training Started')

    G_losses = []
    D_losses = []
    iters = 0
    total_time = 0.0

    for epoch in range(num_epochs):

        for i, data in enumerate(dataloader, 0):

            start_time = time.time()
            batch_size = data[0].size(0)
            # Discriminator
            accuracy, errD, D_x, D_G_z1, fake = train_disc(
                disc, gen, data, device, batch_size, real_label, nz,
                fake_label, criterion_src, criterion_cls, optimD
                )

            # Generator
            errG, D_G_z2 = train_gen(
                disc, gen, real_label, fake, batch_size,
                optimG, criterion_src, criterion_cls
                )

            G_losses.append(errG)
            D_losses.append(errD)

            if (i == len(dataloader)-1):
                fake = gen(test_noise)
                utils.save_image(fake, f'{outdir}images/epoch_{epoch}.png')

            iters += 1

            el_time = time.time() - start_time
            total_time += el_time

            if i % 50 == 0:
                print('[%d/%d][%d/%d]  Loss_D: %.4f  Loss_G: %.4f Accuracy: %.2f  D(x): %.4f  D(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader), errD, errG, accuracy, D_x, D_G_z1, D_G_z2))
        
        # Save latest model state only
        torch.save(gen.state_dict(), f'{outdir}gen.pt')
        torch.save(disc.state_dict(), f'{outdir}disc.pt')

        print(f'Epoch [{epoch}/{num_epochs}] complete\tTime Elapsed : {total_time : .1f} s')
    
    print(f'Finished Training {num_epochs} epochs in {total_time : .1f} seconds')
    return G_losses, D_losses

G_losses, D_losses = train(
    args.num_epochs, dataloader, disc, gen, device, 
    real_label, fake_label, criterion_src, criterion_cls, 
    optimD, optimG, test_noise, args.outdir, args.nz
    )

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

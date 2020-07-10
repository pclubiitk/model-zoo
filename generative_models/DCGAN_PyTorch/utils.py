import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
def compare_img(data,fake):
  img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True).cpu()
  img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True).cpu()
  plt.imshow(img_grid_fake.permute(1, 2, 0))
  plt.imshow(img_grid_real.permute(1, 2, 0))
  
def plot_loss(G_losses,D_losses):
  plt.figure(figsize=(10,5))
  plt.title("Generator and Discriminator Loss During Training")
  plt.plot(G_losses,label="G")
  plt.plot(D_losses,label="D")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()  

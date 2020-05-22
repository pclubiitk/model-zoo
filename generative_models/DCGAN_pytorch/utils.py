import numpy as np
import torchvision
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def animation(img_list):
  fig = plt.figure(figsize=(8,8))
  plt.axis("off")
  ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
  ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
  HTML(ani.to_jshtml())
  
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

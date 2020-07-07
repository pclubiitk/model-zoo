import numpy as np
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import imageio
import os

def plot_loss(g_loss,d_loss,outdir):

    plt.plot(g_loss, label='Generator_Losses')
    plt.plot(d_loss, label='Discriminator Losses')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(outdir,'loss.png'))
    return
to_image = transforms.Compose([transforms.ToPILImage(),transforms.Resize((500,500))])

def make_gif(generator,images,outdir):

    imgs = [np.array(to_image(i)) for i in images]
    imageio.mimsave(outdir+"progress.gif", imgs)
    return

def show_generator(generator,noise):

    to_image(torchvision.utils.make_grid(generator(noise).reshape(-1,1,28,28).cpu().detach()))
    return

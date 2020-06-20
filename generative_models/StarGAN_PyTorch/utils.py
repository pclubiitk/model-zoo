import torch
import matplotlib.pyplot as plt
from main import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plotter(g_losses, d_losses):
    """
    Args:
        g_losses (list): List of iteration-wise generator losses.
        d_losses (list): List of iteration-wise discriminator losses.
    """
    plt.plot(g_losses,label='G')
    plt.plot(d_losses,label='D')
    plt.legend()
    plt.show()

def evaluate(idx, attr):
    """
    Args:
        idx (int): Index of the image from dataset which you want to translate.
        attr (list): Pass a list with length=c_dims, to what you want to translate your image to.
            Example: [0,0,1,0,1]
    """
    D_.eval()
    G_.eval() # Setting the models to eval mode.
    attr=torch.tensor(attr)
    img, lbl=dataset[idx]
    plt.imshow(im1.squeeze().numpy().transpose((1,2,0))) # Plotting original image
    sample=G_(im1.unsqueeze(0),attr.to(device))
    plt.imshow(sample.squeeze().detach().cpu().numpy().transpose((1,2,0)))
    plt.show()
    print('Inital labels: {lbl} , Translated labels: {tst}'.format(lbl=lbl,tst=attr))
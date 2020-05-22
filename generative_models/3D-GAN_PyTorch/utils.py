import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def voxel_plot(directory,threshold):
    """
    Args:
        directory (string): Directory in which you want image to be saved.
                    For example, '/directory' -> '/directory/voxel_T={threshold}.png'
        threshold (float): Should lie between 0 and 1. Above this value voxels are activated.
    """
    evalArray=torch.normal(torch.zeros(1, 200), 
                                torch.ones(1, 200) * .33).to(device)
    evalArray=G_(evalArray)
    evalArray[evalArray>T]=True
    evalArray[evalArray<T]=False
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(evalArray.squeeze().detach().cpu().numpy(),facecolors='red')
    fig.savefig(directory+'/voxel_T={0}.png'.format(threshold))
    plt.show()

def loss_plot(G_losses,D_losses):
    """
    Args:
        G_losses (list): List containing generator losses, with arbitrary scale.
        D_losses (list): List containing discriminator losses, with arbitrary scale.
    """
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.savefig("losses.png")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
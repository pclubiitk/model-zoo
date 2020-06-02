import torch.nn as nn
import torch

def weights_init(m):
    """ Parameter initialization. """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def compute_acc(preds, labels):
    """ Calculate percentage accuracy of the classes
        predicted by the discriminator."""
    correct = 0.0
    preds = torch.max(preds, 1)[1]
    correct = preds.eq(labels.data).sum()
    acc = (correct * 100.0)/ len(labels.data)
    return acc

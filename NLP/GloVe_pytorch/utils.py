import torch
import torch.nn.functional as F

def weight_function(x, x_max, alpha):
    fx = (x/x_max)**alpha
    fx = torch.min(fx, torch.ones_like(fx))
    return fx

def weighted_MSE_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss)

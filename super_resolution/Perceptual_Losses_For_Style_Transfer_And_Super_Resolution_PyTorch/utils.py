import torch
import numpy as np
from torch.nn.functional import mse_loss


def fr_loss(a_C, a_G):
    """Calculate Feature Reconstruction Loss from activations of a layer of Target and Generated Image"""
    loss = mse_loss(a_C, a_G, reduction='mean')
    return loss


def sr_loss(a_S, a_G):
    """Calculate Style Reconstruction Loss from activations of a layer of Style and Generated Image"""
    m, c_S, h_S, w_S = a_S.shape
    m, c_G, h_G, w_G = a_G.shape
    a_S = a_S.view(m, c_S, -1)
    a_G = a_G.view(m, c_G, -1)

    gram_S = (1 / (c_S * h_S * w_S)) * torch.bmm(a_S, a_S.transpose(1, 2))
    gram_G = (1 / (c_G * h_G * w_G)) * torch.bmm(a_G, a_G.transpose(1, 2))
    loss = mse_loss(gram_S, gram_G)
    return loss


def tvr_loss(y):
    """Calculate Total Variation Regularization Loss of Generated Image"""
    loss = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(
        torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    )
    return loss


def tensor_to_image(image):
    return (
        (
            (
                image * np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
                + np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
            ).transpose(0, 2, 3, 1)
            * 255.0
        )
        .clip(0, 255)
        .astype(np.uint8)
    )

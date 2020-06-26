import numpy as np
import torch


def noise_sample(batch_size, disc_c, con_c, noise):

    idx = np.random.randint(10, size=batch_size)
    c = np.zeros((batch_size, 10))
    c[range(batch_size), idx] = 1.0
    disc_c.data.copy_(torch.Tensor(c))
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, disc_c, con_c], 1).view(-1, 74)

    return z, idx


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
            (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

import torch
import torch.nn as nn
from torchvision import models
import torch.utils.model_zoo as model_zoo

def DisLoss(discriminator, imgs, mean, genImgs, real, fake, loss):
    logits, uncondLogits = discriminator(imgs, mean)
    realLoss = loss(logits, real) + loss(uncondLogits, real)

    logits, uncondLogits = discriminator(torch.roll(imgs, 1, 0), mean)
    wrongLoss = loss(logits, fake) + loss(uncondLogits, real)

    logits, uncondLogits = discriminator(genImgs.detach(), mean)
    fakeLoss = loss(logits, fake) + loss(uncondLogits, fake)

    return realLoss + wrongLoss + fakeLoss

def compute_inception_score(predictions, num_splits=1):
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def KLloss(mu, logvar):
    return -torch.mean(mu**2 + logvar.exp() - 1 - logvar)/2

class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        state_dict = \
            model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)

    def forward(self, input):
        x = input * 0.5 + 0.5
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        x = self.model(x)
        x = nn.Softmax()(x)
        return x
  
class custom_loss(nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels do not divide 2!"
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])
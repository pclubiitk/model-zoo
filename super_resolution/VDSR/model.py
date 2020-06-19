import torch
import torch.nn as nn
from math import sqrt

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.layer = self.make_layer(18)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2./n))

    def make_layer(self, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(64, 64, 3, 1, 1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, input):
        residual = input
        out = self.relu(self.conv1(input))
        out = self.layer(out)
        out = self.conv2(out)
        out = torch.add(out, residual)
        return out

# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init


class Image_Discriminator(nn.Module):
    def __init__(self, channel=3):
        super(Image_Discriminator, self).__init__()
        self.n = 64
        self.model = nn.Sequential(
            nn.Conv2d(channel, self.n, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n, self.n*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n*2, self.n * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n*4, self.n * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n*8, 1, 6, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input).view(-1,1).squeeze(1)


class Video_Discriminator(nn.Module):
    def __init__(self, T=16, channel=3):
        """
        input.shape: (channel, T, 96, 96)
        output.shape:
        """
        super(Video_Discriminator, self).__init__()
        self.n = 64
        self.model=nn.Sequential(
            nn.Conv3d(channel, self.n, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(self.n, self.n*2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.n*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(self.n*2, self.n * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.n * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(self.n*4, self.n * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(self.n * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        self.fc = nn.Linear(int((self.n*8)*(T/16)*6*6), 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        output = self.model(input)
        output=self.fc(output.view(output.size(0), -1))
        return self.sig(output).view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, channel=3, z_len=60):
        super(Generator, self).__init__()
        self.n = 64
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_len, self.n*8, 6, 1, bias=False),
            nn.BatchNorm2d(self.n*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.n*8, self.n * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.n*4 , self.n * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.n*2, self.n, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.n, channel, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        return self.model(input)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, gpu=True, dropout = 0):
        super(GRU, self).__init__()
        output_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(output_size, affine=False)
        self.gpu = gpu

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if self.gpu == True:
            self.hidden = self.hidden.cuda()

    def initWeight(self, init_forget_bias=1):
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(params)
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant_(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant_(b_hr, init_forget_bias)
            else:
                init.constant_(params, 0)

    def forward(self, inputs, n_frames):
        outputs = []
        for i in range(n_frames):
            self.hidden = self.gru(inputs, self.hidden)
            inputs = self.linear(self.hidden)
            outputs.append(inputs)
        outputs = [ self.bn(elm) for elm in outputs ]
        outputs = torch.stack(outputs)
        return outputs


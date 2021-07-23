import torch
import torch.nn as nn


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_pooling=True, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = max_pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        skip_connection = out

        if self.max_pooling:
            out = self.maxpool(out)

        return out, skip_connection


class Upsampling_Block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv1 = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=3, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_input):
        # out = self.upsample(x)
        out = self.conv_transpose(x)

        # crop skip_input to match input dim
        start_h = (skip_input.shape[2] - out.shape[2]) // 2
        start_w = (skip_input.shape[3] - out.shape[3]) // 2
        end_h, end_w = start_h + out.shape[2], start_w + out.shape[3]
        skip_input = skip_input[:, :, start_h:end_h, start_w:end_w]

        out = torch.cat((skip_input, out), dim=1)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))

        return out


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, features=64, padding=0):
        super().__init__()
        self.cb1 = Conv_Block(in_channels, features, padding=padding)
        self.cb2 = Conv_Block(features, features * 2, padding=padding)
        self.cb3 = Conv_Block(features * 2, features * 4, padding=padding)
        self.cb4 = Conv_Block(features * 4, features * 8, padding=padding)
        self.cb5 = Conv_Block(
            features * 8, features * 16, max_pooling=False, padding=padding
        )

        self.ub1 = Upsampling_Block(features * 16, features * 8, padding=padding)
        self.ub2 = Upsampling_Block(features * 8, features * 4, padding=padding)
        self.ub3 = Upsampling_Block(features * 4, features * 2, padding=padding)
        self.ub4 = Upsampling_Block(features * 2, features, padding=padding)
        self.conv1x1 = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x):
        out, s1 = self.cb1(x)
        out, s2 = self.cb2(out)
        out, s3 = self.cb3(out)
        out, s4 = self.cb4(out)
        out, _ = self.cb5(out)

        out = self.ub4(self.ub3(self.ub2(self.ub1(out, s4), s3), s2), s1)
        out = self.conv1x1(out)
        return out

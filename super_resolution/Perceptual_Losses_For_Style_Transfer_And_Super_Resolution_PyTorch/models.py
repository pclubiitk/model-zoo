import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode='reflect',
            bias=False,
        )
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in1(self.conv(x)))
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False
        )
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False
        )
        self.in2 = nn.InstanceNorm2d(out_channels, affine=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=stride, mode="nearest")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode='reflect'
        )

    def forward(self, x):
        out = self.conv(self.up(x))
        return out


class StyleTransferNet(nn.Module):
    def __init__(self, in_channels=3, init_features=32, num_residuals=5):
        super().__init__()
        features = init_features
        self.conv1 = ConvBlock(
            in_channels, features, kernel_size=9, stride=1, padding=4
        )
        self.conv2 = ConvBlock(
            features, features * 2, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = ConvBlock(
            features * 2, features * 4, kernel_size=3, stride=2, padding=1
        )

        self.residuals = []
        for _ in range(num_residuals):
            self.residuals.append(ResidualBlock(features * 4, features * 4))
        self.residuals = nn.Sequential(*self.residuals)

        self.up1 = UpsamplingBlock(
            features * 4, features * 2, kernel_size=3, stride=2, padding=1
        )
        self.up2 = UpsamplingBlock(
            features * 2, features, kernel_size=3, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            features, in_channels, kernel_size=9, stride=1, padding=4, padding_mode='reflect', bias=False
        )

    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.residuals(out)
        out = self.up2(self.up1(out))
        out = self.conv4(out)
        return out


class SuperResolutionNet(nn.Module):
    def __init__(
        self, in_channels=3, init_features=64, num_residuals=4, upscale_factor=4
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        features = init_features

        self.conv1 = ConvBlock(
            in_channels, features, kernel_size=9, stride=1, padding=4
        )
        self.residuals = []
        for _ in range(num_residuals):
            self.residuals.append(ResidualBlock(features, features))
        self.residuals = nn.Sequential(*self.residuals)

        self.up1 = UpsamplingBlock(
            features, features, kernel_size=3, stride=2, padding=1
        )
        self.up2 = UpsamplingBlock(
            features, features, kernel_size=3, stride=2, padding=1
        )
        self.up3 = UpsamplingBlock(
            features, features, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            features, in_channels, kernel_size=9, stride=1, padding=4, padding_mode='reflect', bias=False
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.residuals(out)
        out = self.up2(self.up1(out))

        if self.upscale_factor == 8:
            out = self.up3(out)

        out = self.conv2(out)
        return out


class LossNet_VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg16 = models.vgg16(pretrained=True).features
        for parameters in self.vgg16.parameters():
            parameters.requires_grad = False

        self.relu1_2 = nn.Sequential()
        self.relu2_2 = nn.Sequential()
        self.relu3_3 = nn.Sequential()
        self.relu4_3 = nn.Sequential()

        for i in range(0, 4):
            self.relu1_2.add_module("mod_" + str(i), self.vgg16[i])
        for i in range(4, 9):
            self.relu2_2.add_module("mod_" + str(i), self.vgg16[i])
        for i in range(9, 16):
            self.relu3_3.add_module("mod_" + str(i), self.vgg16[i])
        for i in range(16, 23):
            self.relu4_3.add_module("mod_" + str(i), self.vgg16[i])

    def forward(self, x):
        out = self.relu1_2(x)
        relu1_2 = out
        out = self.relu2_2(out)
        relu2_2 = out
        out = self.relu3_3(out)
        relu3_3 = out
        out = self.relu4_3(out)
        relu4_3 = out

        outputs = (relu1_2, relu2_2, relu3_3, relu4_3)
        return outputs

import torch
import torch.nn as nn


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class UNetPP(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision=True, init_features=32):
        super().__init__()

        features = init_features
        self.cb_00 = Conv_Block(in_channels, features)
        self.cb_10 = Conv_Block(features, features*2)
        self.cb_20 = Conv_Block(features*2, features*4)
        self.cb_30 = Conv_Block(features*4, features*8)
        self.cb_40 = Conv_Block(features*8, features*16)

        self.cb_01 = Conv_Block(features*3, features)
        self.cb_11 = Conv_Block(features*2*3, features*2)
        self.cb_21 = Conv_Block(features*4*3, features*4)
        self.cb_31 = Conv_Block(features*8*3, features*8)

        self.cb_02 = Conv_Block(features*4, features)
        self.cb_12 = Conv_Block(features*2*4, features*2)
        self.cb_22 = Conv_Block(features*4*4, features*4)

        self.cb_03 = Conv_Block(features*5, features)
        self.cb_13 = Conv_Block(features*2*5, features*2)

        self.cb_04 = Conv_Block(features*6, features)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.output1 = nn.Conv2d(32, num_classes, kernel_size=1)
            self.output2 = nn.Conv2d(32, num_classes, kernel_size=1)
            self.output3 = nn.Conv2d(32, num_classes, kernel_size=1)
            self.output4 = nn.Conv2d(32, num_classes, kernel_size=1)

        else:
            self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x_00 = self.cb_00(x)
        x_10 = self.cb_10(self.max_pool(x_00))
        x_20 = self.cb_20(self.max_pool(x_10))
        x_30 = self.cb_30(self.max_pool(x_20))
        x_40 = self.cb_40(self.max_pool(x_30))

        x_01 = self.cb_01(torch.cat([x_00, self.up(x_10)], 1))
        x_11 = self.cb_11(torch.cat([x_10, self.up(x_20)], 1))
        x_21 = self.cb_21(torch.cat([x_20, self.up(x_30)], 1))
        x_31 = self.cb_31(torch.cat([x_30, self.up(x_40)], 1))

        x_02 = self.cb_02(torch.cat([x_00, x_01, self.up(x_11)], 1))
        x_12 = self.cb_12(torch.cat([x_10, x_11, self.up(x_21)], 1))
        x_22 = self.cb_22(torch.cat([x_20, x_21, self.up(x_31)], 1))

        x_03 = self.cb_03(torch.cat([x_00, x_01, x_02, self.up(x_12)], 1))
        x_13 = self.cb_13(torch.cat([x_10, x_11, x_12, self.up(x_22)], 1))

        x_04 = self.cb_04(torch.cat([x_00, x_01, x_02, x_03, self.up(x_13)], 1))

        if self.deep_supervision:
            out1 = self.output1(x_01)
            out2 = self.output1(x_02)
            out3 = self.output1(x_03)
            out4 = self.output1(x_04)
            return [out1, out2, out3, out4]

        else:
            out = self.output(x_04)
            return out

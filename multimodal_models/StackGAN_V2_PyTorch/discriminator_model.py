import torch
import torch.nn as nn
from helper_functions.ret_image import downscale_16times
from helper_functions.Blocks import Block3x3_leakRelu, downBlock
import helper_functions.config as cfg

class D64(nn.Module):
    def __init__(self, inn_channels):
        super(D64, self).__init__()
        self.inn_channels = inn_channels
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = downscale_16times(ndf, self.inn_channels)
        self.logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((c_code, x_code), 1)
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

class D128(nn.Module):
    def __init__(self, inn_channels):
        super(D128, self).__init__()
        self.inn_channels = inn_channels
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = downscale_16times(ndf, self.inn_channels)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((c_code, x_code), 1)
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

class D256(nn.Module):
    def __init__(self, inn_channels):
        super(D256, self).__init__()
        self.inn_channels = inn_channels
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = downscale_16times(ndf, self.inn_channels)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((c_code, x_code), 1)
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

class D512(nn.Module):
    def __init__(self, inn_channels):
        super(D512, self).__init__()
        self.inn_channels = inn_channels
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = downscale_16times(ndf, self.inn_channels)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        self.img_code_s128_1 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        self.img_code_s128_2 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s128_3 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s128_1(x_code)
        x_code = self.img_code_s128_2(x_code)
        x_code = self.img_code_s128_3(x_code)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((c_code, x_code), 1)
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

class D1024(nn.Module):
    def __init__(self, inn_channels):
        super(D1024, self).__init__()
        self.inn_channels = inn_channels
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = downscale_16times(ndf, self.inn_channels)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        self.img_code_s256 = downBlock(ndf * 64, ndf * 128)
        self.img_code_s256_1 = Block3x3_leakRelu(ndf * 128, ndf * 64)
        self.img_code_s256_2 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        self.img_code_s256_3 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s256_4 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s256(x_code)
        x_code = self.img_code_s256_1(x_code)
        x_code = self.img_code_s256_2(x_code)
        x_code = self.img_code_s256_3(x_code)
        x_code = self.img_code_s256_4(x_code)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((c_code, x_code), 1)
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]
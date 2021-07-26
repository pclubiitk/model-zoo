import torch
import torch.nn as nn
from generator_model import G1, G2
from helper_functions.Blocks import downBlock, Block3x3_leakRelu
from helper_functions.ret_image import Interpolate, condAugmentation
from helper_functions.initial_weights import weights_init
from helper_functions.losses import KLloss, custom_loss
from helper_functions.Blocks import upScale, normalBlock, Residual

import helper_functions.config as cfg

class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh())

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

class G_NET(nn.Module):
    def __init__(self, StageNum, zDim = 100):
        super(G_NET, self).__init__()
        self.zDim = zDim
        self.StageNum = StageNum
        self.gf_dim = cfg.generatorDim
        self.define_module()

    def define_module(self):
        self.ca_net = condAugmentation()
        if self.StageNum == 1:
            self.h_net1 = G1(self.gf_dim * 16, self.zDim)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        elif self.StageNum == 2:
            self.h_net1 = G1(self.gf_dim * 16, self.zDim)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
            self.h_net2 = G2(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        elif self.StageNum == 3:
            self.h_net1 = G1(self.gf_dim * 16, self.zDim)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
            self.h_net2 = G2(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
            self.h_net3 = G2(self.gf_dim // 2)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)
        elif self.StageNum == 4:
            self.h_net1 = G1(self.gf_dim * 16, self.zDim)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
            self.h_net2 = G2(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
            self.h_net3 = G2(self.gf_dim // 2)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)
            self.h_net4 = G2(self.gf_dim // 4, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 8)

    def forward(self, z_code, text_embedding=None):
        c_code, mu, logvar = self.ca_net(text_embedding)
        fake_imgs = []
        if self.StageNum == 1:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        elif self.StageNum == 2:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        elif self.StageNum == 3:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
        elif self.StageNum == 4:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            h_code4 = self.h_net4(h_code3, c_code)
            fake_img4 = self.img_net4(h_code4)
            fake_imgs.append(fake_img4)
        return fake_imgs, mu, logvar

class eval256(nn.Module):
    def __init__(self):
        super(eval256, self).__init__()
        self.df_dim = cfg.discriminatorDim
        self.ef_dim = cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        h_c_code = x_code

        output = self.logits(h_c_code)

        return output.view(-1)
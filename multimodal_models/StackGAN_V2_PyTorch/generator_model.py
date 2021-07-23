import torch
import torch.nn as nn
import helper_functions.config as cfg
from helper_functions.losses import custom_loss
from helper_functions.Blocks import upScale, normalBlock, Residual

class G1(nn.Module):
    def __init__(self, ngf, zDim = 100):
        super(G1, self).__init__()
        self.gf_dim = ngf
        self.in_dim = zDim + cfg.embeddingsDim
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            custom_loss())
        self.upsample1 = upScale(ngf, ngf // 2)
        self.upsample2 = upScale(ngf // 2, ngf // 4)
        self.upsample3 = upScale(ngf // 4, ngf // 8)
        self.upsample4 = upScale(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        in_code = torch.cat((c_code, z_code), 1)
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        return out_code

class G2(nn.Module):
    def __init__(self, ngf, num_residual = 2):
        super(G2, self).__init__()
        self.gf_dim = ngf

        self.ef_dim = cfg.embeddingsDim
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        self.jointConv = normalBlock(self.gf_dim + self.ef_dim, self.gf_dim)
        self.residual = self._make_layer(Residual, self.gf_dim)
        self.upsample = upScale(self.gf_dim, self.gf_dim // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((c_code, h_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.upsample(out_code)
        return out_code
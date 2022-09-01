import torch
import configparser
import torch.nn as nn
from models import *
from torch import nn, optim
from utils.util import get_model
from utils.util import optical_flow_warping


class ModelBase(nn.Module):
    def __init__(self, model, optimizer):
        super(ModelBase, self).__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)


class CodecNet(nn.Module):

    def __init__(self, cfg, device, flg='ALL'):
        super(CodecNet, self).__init__()
        self.cp = configparser.ConfigParser()
        self.cfg = cfg
        self.optimizers = []
        self.flg = flg
        self.motion_estimate_net = self.load('ME', device)
        self.motion_compensate_net = self.load('MC', device)
        self.motion_encoder = self.load('MVE', device)
        self.motion_decoder = self.load('MVE', device)
        self.residual_encoder = self.load('REE', device)
        self.residual_decoder = self.load('RED', device)
        self.loss = 0
        self.mse_loss = torch.nn.MSELoss(reduction='mean')


    def load(self, model_title, device):
        self.cp.read(self.cfg)
        pretrain = self.cp.get(model_title, 'pretrain')
        name = self.cp.get(model_title, 'name')
        optimizer = self.cp.get(model_title, 'optimizer')
        lr = float(self.cp.get(model_title, 'lr'))

        if model_title == 'ME' or model_title == 'MC':
            model = find_model_using_name(name)()
        else:
            model = find_model_using_name(name)(device)
        model = get_model(model, device, pretrain)

        if optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        self.optimizers.append(optimizer)

        return ModelBase(model, optimizer)


    def forward(self, ref, cur):
        motion_vector, _ = self.motion_estimate_net.model(ref, cur)

        mv_latent, _ = self.motion_encoder(motion_vector / 255)
        mv_quantized, _ = self.motion_decoder.model(mv_latent)
        mv_quantized *= 255

        warped = optical_flow_warping(ref, mv_quantized)
        compensate_input = torch.cat([ref, warped, mv_quantized], dim=1)
        compensate_result = self.motion_compensate_net(compensate_input)

        residual = cur - compensate_result
        res_latent, _ = self.residual_encoder(residual / 255)
        res_quantized, _ = self.residual_decoder(res_latent)
        res_quantized *= 255

        self.retrived_frame = compensate_result + res_quantized
        self.current = cur
        self.compensate_result = compensate_result


    def count_loss(self):
        if self.flg == 'ALL':
            mse = self.mse_loss(self.retrived_frame, self.current)
            bpp_mv = self.motion_encoder.model.entropy_model.loss()
            bpp_re = self.residual_encoder.model.entropy_model.loss()
            self.loss = mse + bpp_mv + bpp_re

        if self.flg == 'WORAE':
            mse = self.mse_loss(self.compensate_result, self.current)
            bpp_mv = self.motion_encoder.model.entropy_model.loss()
            self.loss = mse + bpp_mv


    def optimizer_step(self):
        self.loss.backward()
        for optim in self.optimizers:
            optim.zero_grad()
            optim.step()

if __name__ == "__main__":
    gpu_id = 3
    device = torch.device('cuda:{}'.format(gpu_id))
    CodecNet('./parameters.cfg', device)
    print('a')






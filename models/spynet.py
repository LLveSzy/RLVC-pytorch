import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.util import optical_flow_warping


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = self.conv_relu(8, 32)
        self.conv2 = self.conv_relu(32, 64)
        self.conv3 = self.conv_relu(64, 32)
        self.conv4 = self.conv_relu(32, 16)
        self.conv5 = nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3, padding_mode='replicate')

    def forward(self, cur, ref, flow):
        warp = optical_flow_warping(ref, flow)
        x = torch.cat((warp, cur, flow), axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)

    def conv_relu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv2d(feat_in, feat_out, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(inplace=False))


class SpyNet(nn.Module):
    def __init__(self, stage=4):
        super(SpyNet, self).__init__()
        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layers = nn.ModuleList([ConvLayer()]*4)
        self.mse_Loss = torch.nn.MSELoss(reduction='mean')
        self.stage = stage

    def forward(self, cur, ref):
        '''
        :param cur:  current frame
        :param ref:  reference frame
        :return: flow, loss_list
        '''
        loss_list = []
        down_sample_list = []
        for _ in range(4):
            down_sample_list.append([cur, ref])
            cur = self.down_sample(cur)
            ref = self.down_sample(ref)
        flow = torch.zeros(ref.shape[0], 2, ref.shape[2], ref.shape[3]).to(cur.device)

        flg = False
        for i in range(3, -1, -1):
            cur_frame, ref_frame = down_sample_list[i]
            flow = F.upsample(flow, (ref_frame.shape[2], ref_frame.shape[3]), mode='bilinear')
            refined = self.layers[i](cur_frame, ref_frame, flow)
            refined_flow = refined + flow
            refined_frame = optical_flow_warping(ref_frame, refined_flow)
            loss_list.append(self.mse_Loss(refined_frame, cur_frame))
            flow = refined_flow
            # if not flg and i == 4 - self.stage:
            #     flg = True
            #     save1 = cur_frame[0].permute(1, 2, 0).cpu().detach().numpy()
            #     save2 = refined_frame[0].permute(1, 2, 0).cpu().detach().numpy()
            #     save3 = ref_frame[0].permute(1, 2, 0).cpu().detach().numpy()
            #     cv2.imwrite('./outs/res' + str(random.randint(1, 10)) + '.png',
            #                 np.concatenate((save1, save2, save3), axis=1))
        return flow, loss_list

    def predict_recurrent(self, frames):
        '''
        :param frames: [B, T, C, W, H]
        :return: flows [B, T-1, C, W, H]
        '''
        with torch.no_grad():
            for i in range(1, frames.shape[1]):
                if i == 1:
                    flows = self.forward(frames[:, i, ...], frames[:, 0, ...])[0].unsqueeze(0)
                else:
                    flow = self.forward(frames[:, i, ...], frames[:, 0, ...])[0].unsqueeze(0)
                    flows = torch.cat([flows, flow], dim=0)
        return flows.permute(1, 0, 2, 3, 4)  # -> [B, T-1, C, W, H]



if __name__ == '__main__':
    net = SpyNet()
    input_frame = torch.rand(4, 3, 256, 256)
    ref_frame = torch.rand(4, 3, 256, 256)
    flow, losses = net(input_frame, ref_frame)
    print(losses)
import cv2
import torch
import random
import numpy as np

from tqdm import tqdm
from utils.util import *

from models.unet import UNet
from models.spynet import SpyNet
from models.res_ednet import ResynthesisNet, ReanalysisNet
from models.flow_ednet import MvanalysisNet, MvsynthesisNet

from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.vimeo_dataset import VimeoDataset, VimeoGroupDataset


if __name__ == "__main__":
    gpu_id = 5
    device = torch.device('cuda:{}'.format(gpu_id))
    spynet_pretrain = f'./checkpoints/finetuning1.pth'
    mv_encoder_pretrain = f'./checkpoints/motion_encoder_e.pth'
    mv_decoder_pretrain = f'./checkpoints/motion_decoder_e.pth'
    unet_pretrain = f'./checkpoints/unet.pth'
    re_encoder_pretrain = f'./checkpoints/residual_encoder_e.pth'
    re_decoder_pretrain = f'./checkpoints/residual_decoder_e.pth'

    spynet = get_model(SpyNet(), device, spynet_pretrain)
    mv_encoder = get_model(MvanalysisNet(device), device, mv_encoder_pretrain)
    mv_decoder = get_model(MvsynthesisNet(device), device, mv_decoder_pretrain)
    unet = get_model(UNet(8, 3), device, unet_pretrain)
    re_encoder = get_model(ReanalysisNet(device), device, re_encoder_pretrain)
    re_decoder = get_model(ResynthesisNet(device), device, re_decoder_pretrain)

    dataset = VimeoGroupDataset('/data/szy/datasets/vimeo_septuplet/sequences/')
    frames = torch.Tensor(dataset[432]).unsqueeze(0).to(device)
    last = frames[:, 0, ...]

    h1_state = h2_state = h3_state = h4_state = None
    for i in range(1, 3):
        current = frames[:, i, ...]
        flows, _ = spynet(current, last)

        code, h1_state = mv_encoder(flows, h1_state)
        res, _ = mv_decoder(code, h2_state)
        warped = optical_flow_warping(last, res)
        compens_input = torch.cat([last, warped, res], dim=1)
        compens_result = unet(compens_input)
        residual = (current - compens_result)
        re_code, _ = re_encoder(residual, h3_state)
        re_res, _ = re_decoder(re_code, h4_state)
        refined_frames = compens_result + re_res

        code_q = torch.round(code)
        res_q, h2_state = mv_decoder(code_q, h2_state)
        # warp from last frame
        warped_q = optical_flow_warping(last, res_q)
        # motion compensation net forward & get residual
        compens_input_q = torch.cat([last, warped_q, res_q], dim=1)
        compens_result_q = unet(compens_input_q)
        residual_q = (current - compens_result_q)
        # encoding & decoding residuals
        re_code_q, h3_state = re_encoder(residual_q, h3_state)
        re_res_q, h4_state = re_decoder(re_code_q, h4_state)
        refined_frames_q = compens_result_q + re_res_q

        last = refined_frames_q


    print(flows[0].max(), code.max(), code_q.max(), ((res-flows)**2).mean())

    save1 = last[0].permute(1, 2, 0).cpu().detach().numpy()
    save2 = current[0].permute(1, 2, 0).cpu().detach().numpy()
    save3 = refined_frames[0].permute(1, 2, 0).cpu().detach().numpy()
    save4 = re_res[0].permute(1, 2, 0).cpu().detach().numpy()
    cv2.imwrite('./outs/res' + str(random.randint(1, 10)) + '.png',
                np.concatenate((save1, save2, save3, save4), axis=1))



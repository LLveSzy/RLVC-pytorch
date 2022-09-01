import cv2
import math
import torch
import random
import numpy as np
import compressai
import utils.artithmeticcoding

from tqdm import tqdm
from utils.util import *
from utils.evaluations import ms_ssim
from models.unet import UNet
from models.spynet import SpyNet
from models.res_ednet import ResynthesisNet, ReanalysisNet
from models.flow_ednet import MvanalysisNet, MvsynthesisNet

from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.vimeo_dataset import VimeoDataset, VimeoGroupDataset, UVGDataset



if __name__ == "__main__":
    gpu_id = 3
    device = torch.device('cuda:{}'.format(gpu_id))
    spynet_pretrain = f'./checkpoints/spynet_union.pth'
    mv_encoder_pretrain = f'./checkpoints/motion_encoder_union.pth'
    mv_decoder_pretrain = f'./checkpoints/motion_decoder_union.pth'
    unet_pretrain = f'./checkpoints/unet_light1.pth'
    re_encoder_pretrain = f'./checkpoints/residual_encoder_union.pth'
    re_decoder_pretrain = f'./checkpoints/residual_decoder_union.pth'

    spynet = get_model(SpyNet(), device, spynet_pretrain)
    mv_encoder = get_model(MvanalysisNet(device), device, mv_encoder_pretrain)
    mv_decoder = get_model(MvsynthesisNet(device), device, mv_decoder_pretrain)
    unet = get_model(UNet(8, 3), device, unet_pretrain)
    re_encoder = get_model(ReanalysisNet(device), device, re_encoder_pretrain)
    re_decoder = get_model(ResynthesisNet(device), device, re_decoder_pretrain)

    dataset = VimeoGroupDataset('/data/szy/datasets/vimeo_septuplet/sequences/')
    # dataset = UVGDataset('/data/szy/datasets/videos_crop/Jockey_1920x1024_120fps_420_8bit_YUV/H265L26/')
    frames = torch.Tensor(dataset[344]).unsqueeze(0).to(device)
    last = frames[:, 0, ...]

    h1_state = h2_state = h3_state = h4_state = None
    entropy_model = compressai.entropy_models.entropy_models.EntropyBottleneck(128).to(device)
    entropy_model.update()
    mse_loss = torch.nn.MSELoss(reduction='mean')
    psnrs = []
    bpps = []
    ms = []

    height, width = frames.shape[3:]
    mv_encoder.entropy_model.update()
    re_encoder.entropy_model.update()
    h1_stated = h2_state = h3_state = h4_state = None
    for i in range(1, 2):
        h1_state = h2_state = h3_state = h4_state = None
        current = frames[:, i, ...]
        flows, _ = spynet(current, last)
        code, h1_state = mv_encoder(flows/255, h1_state)
        res, _ = mv_decoder(code)
        res *= 255
        warped = optical_flow_warping(last, res)
        compens_input = torch.cat([last, warped, res], dim=1)
        compens_result = unet(compens_input)
        residual = (current - compens_result)
        re_code, _ = re_encoder(residual / 255)
        re_res, _ = re_decoder(re_code)
        re_res *= 255
        refined_frames = compens_result + re_res

        code_q = torch.round(code)
        mv_string = mv_encoder.entropy_model.compress(code_q)
        # out, liklihood = entropy_model(code_q)

        # temp = torch.nn.functional.interpolate(code_q, scale_factor=0.5)
        # code_q = torch.nn.functiondal.interpolate(temp, scale_factor=2)

        # mv_string = entropy_model.compress(code_q)
        res_q, h2_state = mv_decoder(code_q, h2_state)
        res_q *= 255
        # warp from last frame
        warped_q = optical_flow_warping(last, res_q)
        # motion compensation net forward & get residual
        compens_input_q = torch.cat([last, warped_q, res_q], dim=1)
        compens_result_q = unet(compens_input_q)
        residual_q = (current - compens_result_q)
        # encoding & decoding residuals
        re_code_q, h3_state = re_encoder(residual_q / 255, h3_state)
        re_code_q = torch.round(re_code_q)

        # h, w = re_code_q.shape[2:]
        # temp = torch.nn.functional.interpolate(re_code_q, size=(h-6, w-8))
        # re_code_q = torch.nn.functional.interpolate(temp, size=(h, w))

        res_string = re_encoder.entropy_model.compress(re_code_q)

        re_res_q, h4_state = re_decoder(re_code_q, h4_state)
        re_res_q *= 255
        refined_frames_q = compens_result_q + re_res_q

        ms.append(ms_ssim(refined_frames_q, current).item())
        last = refined_frames_q

        mse = mse_loss(current / 255., refined_frames_q / 255.)
        psnr = 10.0 * torch.log10(1.0 / mse)
        psnrs.append(psnr.item())

        bpps.append((2 + len(mv_string[0]) + len(res_string[0])) / height / width )
        # bpps.append(mv_string.item())


    print(psnrs, bpps, ms)
    # print(flows[0].max(), code.max(), code_q.max(), ((res-flows)**2).mean())

    save1 = current[0].permute(1, 2, 0).cpu().detach().numpy()
    save2 = compens_result_q[0].permute(1, 2, 0).cpu().detach().numpy()
    save3 = residual_q[0].permute(1, 2, 0).cpu().detach().numpy()
    save4 = last[0].permute(1, 2, 0).cpu().detach().numpy()
    cv2.imwrite('./outs/res' + str(random.randint(11, 20)) + '.png',
                np.concatenate((save1, save2, save3, save4), axis=1))



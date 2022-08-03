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
    flows = spynet.predict_recurrent(frames)
    code = mv_encoder(flows)
    code_q = torch.round(code)
    res = mv_decoder(code)
    res_q = mv_decoder(code_q)
    print(flows[0].max(), code.max(), code_q.max(), ((res-flows)**2).mean())
    # print(frames.shape, res.shape)
    warped = optical_flow_warping(frames[:, 0, ...], res[:, 3, ...])
    warped_q = optical_flow_warping(frames[:, 0, ...], res_q[:, 3, ...])

    save1 = frames[0][0].permute(1, 2, 0).cpu().detach().numpy()
    save2 = warped[0].permute(1, 2, 0).cpu().detach().numpy()
    save3 = warped_q[0].permute(1, 2, 0).cpu().detach().numpy()
    save4 = frames[0][4].permute(1, 2, 0).cpu().detach().numpy()
    cv2.imwrite('./outs/res' + str(random.randint(1, 10)) + '.png',
                np.concatenate((save1, save2, save3, save4), axis=1))

    # mark last three sizes
    l3_shapes = frames.shape[2:]
    l2_shapes = flows.shape[2:]
    # get key frames
    first_frames = frames[:, 0, ...].unsqueeze(1)
    first_frames = torch.repeat_interleave(first_frames, repeats=flows.shape[1], dim=1) \
        .view(-1, *l3_shapes).to(device)
    warped = optical_flow_warping(first_frames, res.squeeze(0))
    warped_q = optical_flow_warping(first_frames, res_q.squeeze(0))

    # warp from first frame
    res_for_warp = res_q.view(-1, *l2_shapes)
    warped = warped_q.view(-1, *l3_shapes)
    # motion compensation net forward
    compens_input = torch.cat([first_frames, warped, res_for_warp], dim=1)
    compens_result = unet(compens_input)
    # encoding & decoding residuals
    frame_except_first = frames[:, 1:, ...].reshape(-1, *l3_shapes).to(device)
    residual = (frame_except_first - compens_result).view(1, -1, *l3_shapes)
    re_code = re_encoder(residual)
    re_res = re_decoder(re_code)
    re_res_q = re_decoder(torch.round(re_code))

    refined_frames = (re_res + compens_result).view(1, -1, *l3_shapes)
    refined_frames_q = (re_res_q + compens_result).view(1, -1, *l3_shapes)
    save1 = frames[0][5].permute(1, 2, 0).cpu().detach().numpy()
    save2 = refined_frames[0][4].permute(1, 2, 0).cpu().detach().numpy()
    save3 = re_res[0][4].permute(1, 2, 0).cpu().detach().numpy()
    cv2.imwrite('./outs/res' + str(random.randint(10, 20)) + '.png',
                np.concatenate((save1, save2, save3), axis=1))


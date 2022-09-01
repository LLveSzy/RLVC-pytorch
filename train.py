import cv2
import math
import torch
import random
import numpy as np
import torch.nn as nn
import compressai
from tqdm import tqdm
from utils.util import *
from torch import nn, optim
from models.unet import UNet
from models.spynet import SpyNet
from utils.visualize import Visual
from utils.util import optical_flow_warping
from models.flow_ednet import MvanalysisNet, MvsynthesisNet
from models.res_ednet import ReanalysisNet, ResynthesisNet

from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.vimeo_dataset import VimeoDataset, VimeoGroupDataset



if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    gpu_id = 3
    batch_size = 8
    lr = 1e-6
    epochs = 20
    device = torch.device('cuda:{}'.format(gpu_id))

    v = Visual('exp')

    spynet_pretrain = f'./checkpoints/spynet_union.pth'
    mv_encoder_pretrain = f'./checkpoints/motion_encoder_d255.pth'
    mv_decoder_pretrain = f'./checkpoints/motion_decoder_d255.pth'
    unet_pretrain = f'./checkpoints/unet_light1.pth'
    re_encoder_pretrain = f'./checkpoints/residual_encoder_d255.pth'
    re_decoder_pretrain = f'./checkpoints/residual_decoder_d255.pth'

    spynet = get_model(SpyNet(), device, spynet_pretrain)
    mv_encoder = get_model(MvanalysisNet(device), device, mv_encoder_pretrain)
    mv_decoder = get_model(MvsynthesisNet(device), device, mv_decoder_pretrain)
    unet = get_model(UNet(8, 3), device, unet_pretrain)
    #
    # spynet.requires_grad = False
    # unet.requires_grad = False
    # mv_encoder.requires_grad = False
    # mv_decoder.requires_grad = False
####################################################################################################################
    re_encoder = get_model(ReanalysisNet(device), device, re_encoder_pretrain)
    re_decoder = get_model(ResynthesisNet(device), device, re_decoder_pretrain)
    # re_encoder = get_model(ReanalysisNet(device), device)
    # re_decoder = get_model(ResynthesisNet(device), device)

    # optim_list = optimizer_factory(lr, *[re_encoder, re_decoder, unet, mv_encoder, mv_decoder, spynet])
    aux_parameters = set(p for n, p in mv_encoder.named_parameters() if n.endswith(".quantiles"))
    aux_optimizer_mv = optim.SGD(aux_parameters, lr=1e-4, weight_decay=1e-7)
    aux_parameters = set(p for n, p in re_encoder.named_parameters() if n.endswith(".quantiles"))
    aux_optimizer_re = optim.SGD(aux_parameters, lr=1e-4, weight_decay=1e-7)
    optim_list = optimizer_factory(lr, *[spynet, mv_encoder, mv_decoder, re_encoder, re_decoder]) + [aux_optimizer_mv, aux_optimizer_re]

    dataset = VimeoGroupDataset('/data/szy/datasets/vimeo_septuplet/sequences/')
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_train = len(dataset)//20
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:n_train])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    mse_Loss = torch.nn.MSELoss(reduction='mean')

    height, width = dataset[0].shape[2:]
    frac = height * width * batch_size
    global_step = 1
    # re_encoder.entropy_model.update()
    try:
        for epoch in range(1, epochs):
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', ncols=112) as pbar:
                for frames in train_dataloader:
                    loss = 0
                    frames = frames.to(device)
                    last = frames[:, 0, ...]
                    h1_state = h2_state = h3_state = h4_state = None
                    for i in range(1, frames.shape[1]):
                        current = frames[:, i, ...]
                        flows, optical_loss = spynet(current, last)
                        optical_loss = optical_loss[-1]

                        code, h1_state = mv_encoder(flows/255, h1_state)
                        res, h2_state = mv_decoder(code, h2_state)
                        res *= 255
                        mv_ae_loss = log_likelihood(mv_encoder.likelihood, frac) #+ mse_Loss(flows, res)
                        # warp from last frame
                        loss2 = mv_encoder.entropy_model.loss()
                        warped = optical_flow_warping(last, res)
                        # motion compensation net forward & get residual
                        compens_input = torch.cat([last, warped, res], dim=1)
                        compens_result = unet(compens_input)
                        # mc_loss = mse_Loss(current/255, compens_result/255)
                        # psnr = -10.0 * torch.log10(1.0 / mc_loss)
                        # encoding & decoding residuals
                        residual = (current - compens_result)
                        re_code, h3_state = re_encoder(residual / 255, h3_state)
                        re_res, h4_state = re_decoder(re_code, h4_state)
                        re_res *= 255
                        res_ae_loss = log_likelihood(re_encoder.likelihood, frac)  # + mse_Loss(residual, re_res)
                        loss3 = re_encoder.entropy_model.loss()
                        # retrieve frame
                        retrieval_frames = compens_result + re_res
                        retrieval_mse = mse_Loss(retrieval_frames/255., current/255.)
                        retrieval_loss = -10.0 * torch.log10(1.0 / retrieval_mse)

                        last = current

                        rae_loss = res_ae_loss# - loss3
                        mae_loss = mv_ae_loss# - loss2
                        # mc_loss = psnr
                        # if i == 1:
                        loss += 100 * (rae_loss + mae_loss)
                        loss += retrieval_loss


                    # re_string = re_encoder.entropy_model.compress(re_code)
                    # bpp = len(re_string[0]) / width / height / current.shape[0]
                    bpp = res_ae_loss

                    # loss /= (frames.shape[1] - 1)
                    loss.backward()
                    optimizer_step(optim_list)
                    pbar.set_postfix(**{'loss (batch)': format(loss.item(), '.3f'),
                                        'bpp': format(bpp, '.3f'),
                                        'max re_res': format(re_res.max(), '.1f')})
                    pbar.update(frames.shape[0])

                    # v.draw_pics(global_step, [frames[:, -2, ...], residual, re_res, retrieval_frames])
                    # v.draw_pics(global_step, [frames[:, -2, ...], current])

                    global_step += 1
                    save1 = frames[:, -2, ...][0].permute(1, 2, 0).cpu().detach().numpy()
                    save2 = compens_result[0].permute(1, 2, 0).cpu().detach().numpy()
                    save3 = retrieval_frames[0].permute(1, 2, 0).cpu().detach().numpy()
                    save4 = current[0].permute(1, 2, 0).cpu().detach().numpy()
                    cv2.imwrite('./outs/res' + str(random.randint(1, 10)) + '.png',
                                np.concatenate((save1, save2, save3, save4), axis=1))

                    # save1 = frames[:, -2, ...][0].permute(1, 2, 0).cpu().detach().numpy()
                    # save2 = residual[0].permute(1, 2, 0).cpu().detach().numpy()
                    # save3 = re_res[0].permute(1, 2, 0).cpu().detach().numpy()
                    # save4 = retrieval_frames[0].permute(1, 2, 0).cpu().detach().numpy()
                    # cv2.imwrite('./outs/res' + str(random.randint(1, 10)) + '.png',
                    #             np.concatenate((save1, save2, save3, save4), axis=1))

        save_checkpoint(re_encoder, 'residual_encoder_union')
        save_checkpoint(re_decoder, 'residual_decoder_union')
        save_checkpoint(spynet, 'spynet_union')
        save_checkpoint(unet, 'unet_union')
        save_checkpoint(mv_encoder, 'motion_encoder_union')
        save_checkpoint(mv_decoder, 'motion_decoder_union')
    except KeyboardInterrupt:
        save_checkpoint(re_encoder, 'residual_encoder_union')
        save_checkpoint(re_decoder, 'residual_decoder_union')
        save_checkpoint(spynet, 'spynet_union')
        save_checkpoint(unet, 'unet_union')
        save_checkpoint(mv_encoder, 'motion_encoder_union')
        save_checkpoint(mv_decoder, 'motion_decoder_union')
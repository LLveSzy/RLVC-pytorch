import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import compressai
from tqdm import tqdm
from utils.util import *

from models.unet import UNet
from models.spynet import SpyNet
from utils.util import optical_flow_warping
from models.flow_ednet import MvanalysisNet, MvsynthesisNet
from models.res_ednet import ReanalysisNet, ResynthesisNet

from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.vimeo_dataset import VimeoDataset, VimeoGroupDataset



if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    gpu_id = 0
    batch_size = 6
    lr = 1e-5
    epochs = 20
    device = torch.device('cuda:{}'.format(gpu_id))
    spynet_pretrain = f'./checkpoints/stage4.pth'
    mv_encoder_pretrain = f'./checkpoints/motion_encoder_e.pth'
    mv_decoder_pretrain = f'./checkpoints/motion_decoder_e.pth'
    unet_pretrain = f'./checkpoints/unet.pth'
    re_encoder_pretrain = f'./checkpoints/residual_encoder_e.pth'
    re_decoder_pretrain = f'./checkpoints/residual_decoder_e.pth'

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
    # re_encoder = get_model(ReanalysisNet(device), device, re_encoder_pretrain)
    # re_decoder = get_model(ResynthesisNet(device), device, re_decoder_pretrain)
    re_encoder = get_model(ReanalysisNet(device), device)
    re_decoder = get_model(ResynthesisNet(device), device)

    # optim_list = optimizer_factory(lr, *[re_encoder, re_decoder, unet, mv_encoder, mv_decoder, spynet])
    optim_list = optimizer_factory(lr, *[spynet, unet, mv_encoder, mv_decoder])

    dataset = VimeoGroupDataset('/data/szy/datasets/vimeo_septuplet/sequences/')
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_train = len(dataset)//20
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:n_train])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    mse_Loss = torch.nn.MSELoss(reduction='mean')
    entropy_loss = compressai.entropy_models.entropy_models.EntropyBottleneck(128).to(device)
    try:
        for epoch in range(1, epochs):
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}') as pbar:
                for frames in train_dataloader:
                    loss = 0
                    frames = frames.to(device)
                    last = frames[:, 0, ...]
                    h1_state = h2_state = h3_state = h4_state = None
                    for i in range(1, frames.shape[1]):
                        current = frames[:, i, ...]
                        flows, optical_loss = spynet(current, last)
                        optical_loss = optical_loss[-1]

                        code, h1_state = mv_encoder(flows, h1_state)
                        res, h2_state = mv_decoder(code, h2_state)
                        mv_ae_loss = mse_Loss(flows, res) - 0.1 * entropy_loss(code)[1].mean()
                        # warp from last frame
                        warped = optical_flow_warping(last, res)
                        # motion compensation net forward & get residual
                        compens_input = torch.cat([last, warped, res], dim=1)
                        compens_result = unet(compens_input)
                        mc_loss = mse_Loss(current, compens_result)
                        # encoding & decoding residuals
                        residual = (current - compens_result)
                        re_code, h3_state = re_encoder(residual, h3_state)
                        re_res, h4_state = re_decoder(re_code, h4_state)
                        res_ae_loss = mse_Loss(residual, re_res) - 0.1 * entropy_loss(re_code)[1].mean()
                        retrieval_frames = compens_result + re_res
                        retrieval_loss = mse_Loss(retrieval_frames, current)

                        last = retrieval_frames

                        loss += optical_loss + mc_loss + mv_ae_loss # + res_ae_loss + 10 * retrieval_loss
                    loss /= frames.shape[1]

                    loss.backward()
                    optimizer_step(optim_list)
                    pbar.set_postfix(**{'loss (batch)': format(loss.item(), '.5f')})
                    pbar.update(frames.shape[0])

                # save1 = cur[0].permute(1, 2, 0).cpu().detach().numpy()
                    # save2 = pre[0].permute(1, 2, 0).cpu().detach().numpy()
                    # save3 = ref[0].permute(1, 2, 0).cpu().detach().numpy()
                    # cv2.imwrite('./outs/res' + str(random.randint(1, 10)) + '.png',
                    #             np.concatenate((save1, save2, save3), axis=1))

        save_checkpoint(re_encoder, 'residual_encoder_union')
        save_checkpoint(re_decoder, 'residual_decoder_union')
        save_checkpoint(spynet, 'spynet_union')
        save_checkpoint(unet, 'unet_union')
        save_checkpoint(mv_encoder, 'mv_encoder_union')
        save_checkpoint(mv_decoder, 'mv_decoder_union')
    except KeyboardInterrupt:
        save_checkpoint(re_encoder, 'residual_encoder_union')
        save_checkpoint(re_decoder, 'residual_decoder_union')
        save_checkpoint(spynet, 'spynet_union')
        save_checkpoint(unet, 'unet_union')
        save_checkpoint(mv_encoder, 'mv_encoder_union')
        save_checkpoint(mv_decoder, 'mv_decoder_union')
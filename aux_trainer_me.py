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
from models.flow_ednet import MvanalysisNet, MvsynthesisNet

from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.vimeo_dataset import VimeoDataset, VimeoGroupDataset



if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    gpu_id = 5
    batch_size = 10
    lr = 4e-4
    epochs = 20
    device = torch.device('cuda:{}'.format(gpu_id))
    fixed = False

    spynet_pretrain = f'./checkpoints/finetuning1.pth'
    mv_encoder_pretrain = f'./checkpoints/motion_encoder_e.pth'
    mv_decoder_pretrain = f'./checkpoints/motion_decoder_e.pth'

    spynet = get_model(SpyNet(), device, spynet_pretrain)
    mv_encoder = get_model(MvanalysisNet(device), device, mv_encoder_pretrain)
    mv_decoder = get_model(MvsynthesisNet(device), device, mv_decoder_pretrain)

    optimizer_encoder = torch.optim.Adam(mv_encoder.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_decoder = torch.optim.Adam(mv_decoder.parameters(), lr=lr, weight_decay=1e-4)

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
                    # [batch*frame, channel, width, height] -> # [batch, frame-1 , channel, width, height]
                    flows = spynet.predict_recurrent(frames.to(device))
                    code = mv_encoder(flows)
                    output, likelihood = entropy_loss(code.view(-1, *code.shape[2:]))
                    res = mv_decoder(code)
                    mse = mse_Loss(res, flows)
                    loss = mse - 0.1 * torch.log(likelihood).mean()
                    optimizer_encoder.zero_grad()
                    optimizer_decoder.zero_grad()
                    loss.backward()
                    optimizer_encoder.step()
                    optimizer_decoder.step()
                    pbar.set_postfix(**{'loss (batch)': format(loss.item(), '.5f')})
                    pbar.update(frames.shape[0])

                    # save1 = cur[0].permute(1, 2, 0).cpu().detach().numpy()
                    # save2 = pre[0].permute(1, 2, 0).cpu().detach().numpy()
                    # save3 = ref[0].permute(1, 2, 0).cpu().detach().numpy()
                    # cv2.imwrite('./outs/res' + str(random.randint(1, 10)) + '.png',
                    #             np.concatenate((save1, save2, save3), axis=1))

        save_checkpoint(mv_encoder, 'motion_encoder_e')
        save_checkpoint(mv_decoder, 'motion_decoder_e')
    except KeyboardInterrupt:
        save_checkpoint(mv_encoder, 'motion_encoder_e')
        save_checkpoint(mv_decoder, 'motion_decoder_e')
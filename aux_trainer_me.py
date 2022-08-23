import cv2
import math
import torch
import random
import numpy as np
import torch.nn as nn
import compressai

from tqdm import tqdm
from utils.util import *
from utils.visualize import Visual
from models.unet import UNet
from models.spynet_tst import SpyNet
from models.flow_ednet import MvanalysisNet, MvsynthesisNet

from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.vimeo_dataset import VimeoDataset, VimeoGroupDataset



if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    gpu_id = 5
    batch_size = 10
    lr = 1e-4
    epochs = 20
    device = torch.device('cuda:{}'.format(gpu_id))
    fixed = False

    v = Visual('exp')
    # spynet_pretrain = f'./checkpoints/stage4.pth'
    mv_encoder_pretrain = f'./checkpoints/motion_encoder_d255.pth'
    mv_decoder_pretrain = f'./checkpoints/motion_decoder_d255.pth'

    spynet = get_model(SpyNet(), device)
    mv_encoder = get_model(MvanalysisNet(device), device, mv_encoder_pretrain)
    mv_decoder = get_model(MvsynthesisNet(device), device, mv_decoder_pretrain)

    optimizer_encoder = torch.optim.Adam(mv_encoder.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_decoder = torch.optim.Adam(mv_decoder.parameters(), lr=lr, weight_decay=1e-4)

    dataset = VimeoDataset('/data/szy/datasets/vimeo_septuplet/sequences/')
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_train = len(dataset)//20
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:n_train])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    mse_Loss = torch.nn.MSELoss(reduction='mean')
    entropy_loss = compressai.entropy_models.entropy_models.EntropyBottleneck(128).to(device)
    global_step = 0
    try:
        for epoch in range(1, epochs):
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', ncols=110) as pbar:
                for ref, cur in train_dataloader:
                    # [batch*frame, channel, width, height] -> # [batch, frame-1 , channel, width, height]
                    height, width = ref.shape[2:]
                    ref = ref.to(device)
                    cur = cur.to(device)
                    flows = spynet(cur, ref)
                    code, _ = mv_encoder(flows)
                    output, likelihood = entropy_loss(code)
                    res, _ = mv_decoder(output)
                    warped = optical_flow_warping(ref, res)
                    mse = mse_Loss(res, flows)
                    psnr = -10.0 * torch.log10(1.0 / mse)
                    loss = mse * 0.1 - torch.log(likelihood).sum() / height / width / batch_size / math.log(2)
                    optimizer_encoder.zero_grad()
                    optimizer_decoder.zero_grad()
                    loss.backward()
                    optimizer_encoder.step()
                    optimizer_decoder.step()
                    pbar.set_postfix(**{'loss (batch)': format(loss.item(), '.2f'),
                                        'mean res': format(res.max(), '.1f'),
                                        'mean flo_res': format(flows.max(), '.1f')
                                        })
                    pbar.update(ref.shape[0])

                    v.draw_pics(global_step, [cur, warped, ref])
                    global_step += 1

        save_checkpoint(mv_encoder, 'motion_encoder_d2551')
        save_checkpoint(mv_decoder, 'motion_decoder_d2551')
    except KeyboardInterrupt:
        save_checkpoint(mv_encoder, 'motion_encoder_d2551')
        save_checkpoint(mv_decoder, 'motion_decoder_d2551')
import cv2
import torch
import random
import numpy as np

from tqdm import tqdm
from models.unet import UNet
from models.spynet import SpyNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.util import *
from dataset.vimeo_dataset import VimeoDataset


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    gpu_id = 5
    batch_size = 24
    lr = 1e-3
    epochs = 100
    device = torch.device('cuda:{}'.format(gpu_id))
    fixed = False

    spynet_pretrain = f'./checkpoints/stage4.pth'
    unet_pretrain = f'./checkpoints/unet.pth'

    spynet = get_model(SpyNet(), device, spynet_pretrain)
    unet = get_model(UNet(8, 3), device, unet_pretrain)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, unet.parameters()), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)
    dataset = VimeoDataset('/data/szy/datasets/vimeo_septuplet/sequences/')
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_train = len(dataset)//20
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:n_train])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    mse_Loss = torch.nn.MSELoss(reduction='mean')
    try:
        for epoch in range(1, epochs):
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}') as pbar:
                for cur, ref in train_dataloader:
                    cur = Variable(cur).to(device)
                    ref = Variable(ref).to(device)
                    flow, _ = spynet(cur, ref)
                    warped_frame = optical_flow_warping(ref, flow)

                    unet_input = torch.cat([ref, warped_frame, flow], axis=1)
                    pre = unet(unet_input)
                    loss = mse_Loss(pre, cur)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step(losses[0])
                    pbar.set_postfix(**{'loss (batch)': format(loss.item(), '.5f')})
                    pbar.update(cur.shape[0])

                    save1 = cur[0].permute(1, 2, 0).cpu().detach().numpy()
                    save2 = pre[0].permute(1, 2, 0).cpu().detach().numpy()
                    save3 = ref[0].permute(1, 2, 0).cpu().detach().numpy()
                    cv2.imwrite('./outs/res' + str(random.randint(1, 10)) + '.png',
                                np.concatenate((save1, save2, save3), axis=1))

        save_checkpoint(unet, 'unet')
    except KeyboardInterrupt:
        save_checkpoint(unet, 'unet')



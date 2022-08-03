import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from models.spynet import SpyNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.util import *
from dataset.vimeo_dataset import VimeoDataset


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    gpu_id = 5
    batch_size = 40
    lr = 5e-4
    epochs = 15
    device = torch.device('cuda:{}'.format(gpu_id))
    source = 4
    stage = 4
    fixed = False

    net = SpyNet(stage).to(device)

    if fixed:
        save_checkpoint_name = 'stage' + str(stage)
        checkpoint_name = 'stage' + str(source)
        for name, module in net.named_children():
            if name == 'layers':
                for i in range(stage-2, -1, -1):
                    print(f'{name + str(i)} loaded')
                    module[i].requires_grad = False
    else:
        checkpoint_name = 'finetuning'
        save_checkpoint_name = 'finetuning1'

    pre_trained = f'./checkpoints/{checkpoint_name}.pth'
    pretrain_encoder = torch.load(pre_trained, map_location=device)
    net.load_state_dict(pretrain_encoder)
    print(f'loaded: {checkpoint_name}.pth')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)
    dataset = VimeoDataset('/data/szy/datasets/vimeo_septuplet/sequences/')
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_train = len(dataset)//20
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:n_train])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    try:
        for epoch in range(1, epochs):
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}') as pbar:
                for cur, ref in train_dataloader:
                    cur = Variable(cur).to(device)
                    ref = Variable(ref).to(device)
                    flow, losses = net(cur, ref)

                    optimizer.zero_grad()
                    if fixed:
                        loss = losses[stage-1]
                    else:
                        loss = sum(losses[:stage])
                    loss.backward()
                    optimizer.step()
                    # scheduler.step(losses[0])
                    pbar.set_postfix(**{'loss (batch)': format(loss.item(), '.5f')})
                    pbar.update(cur.shape[0])

        save_checkpoint(net, save_checkpoint_name)
    except KeyboardInterrupt:
        save_checkpoint(net, save_checkpoint_name)



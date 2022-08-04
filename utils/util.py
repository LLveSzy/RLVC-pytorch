import os
import torch
import torch.nn.functional as F


def optical_flow_warping(x, flo, pad_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    pad_mode (optional): ref to https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        "zeros": use 0 for out-of-bound grid locations,
        "border": use border values for out-of-bound grid locations
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    vgrid = vgrid.to(x.device)
    output = F.grid_sample(x, vgrid, padding_mode=pad_mode)

    mask = torch.ones(x.size()).to(x.device)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def save_checkpoint(model, checkpoint_name):
    dir_checkpoint = f'./checkpoints/'
    torch.save(model.state_dict(),
               os.path.join(dir_checkpoint, f'{checkpoint_name}.pth'))
    print(f'Checkpoint {checkpoint_name} saved !')


def get_model(model, device, checkpoint=None):
    if checkpoint:
        pre_trained = checkpoint
        pretrain_encoder = torch.load(pre_trained, map_location=device)
        model.load_state_dict(pretrain_encoder)
    return model.to(device)

def load_statedict(model, checkpoint, device):
    not_match = 0
    state_dict = torch.load(checkpoint, map_location=device)
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            not_match += 1
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)
    if not_match != 0:
        logger.warning('Params not match: ' + str(not_match))
    else:
        logger.info('ALL MATCHED.')
    return own_state


def optimizer_factory(lr, *args):
    optmizer_list = []
    for o in args:
        optmizer_list.append(torch.optim.Adam(o.parameters(), lr=lr, weight_decay=1e-4))
    return optmizer_list


def optimizer_step(optmizer_list):
    for optim in optmizer_list:
        optim.zero_grad()
        optim.step()

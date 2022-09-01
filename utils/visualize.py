import os
import numpy as np
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

class Visual:
    def __init__(self, path):
        self.writer = SummaryWriter(os.path.join('./checkpoints/', path))

    def draw_pics(self, global_step, tensor_list):
        images = []
        for img in tensor_list:
            img = nn.functional.relu(img[0].permute(1, 2, 0)).cpu().detach().numpy()
            img[img > 255] = 255
            images.append(img)
        img_for_display = np.concatenate(tuple(images), axis=0).astype(np.int8)
        self.writer.add_images('results', img_for_display, global_step, dataformats='HWC')
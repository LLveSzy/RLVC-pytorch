import os
import cv2
import torch
import random
import numpy as np
import torch.utils.data as data

from os.path import join
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


class VimeoDataset(data.Dataset):
    def __init__(self, root):
        self.pth = []
        for rt in os.listdir(root):
            for i in os.listdir(join(root, rt)):
                self.pth.append(join(root, rt, i))
        self._transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])

    def __getitem__(self, idx):
        pic_pth = self.pth[idx]
        lst = [i for i in range(1, 6)]
        random.shuffle(lst)
        num1, num2 = lst[:2]
        ref = cv2.imread(join(pic_pth, 'im'+ str(num1) + '.png'))
        cur = cv2.imread(join(pic_pth, 'im'+ str(num1+1) + '.png'))
        ref, cur = torch.Tensor(cur).permute(2, 0, 1).unsqueeze(0), torch.Tensor(ref).permute(2, 0, 1).unsqueeze(0)
        ref, cur = self._transform(torch.cat([ref, cur], dim=0))
        return ref, cur

    def __len__(self):
        return len(self.pth)


class VimeoGroupDataset(data.Dataset):
    def __init__(self, root):
        self.pth = []
        for rt in os.listdir(root):
            for i in os.listdir(join(root, rt)):
                self.pth.append(join(root, rt, i))
        self._transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])

    def __getitem__(self, idx):
        pic_pth = self.pth[idx]
        # start = random.randint(0, 4)
        start = 0
        end = 3
        for i in range(start+1, start+end):
            f = cv2.imread(join(pic_pth, 'im'+ str(i) + '.png'))
            f = torch.Tensor(f).permute(2, 0, 1).unsqueeze(0)
            if i == start+1:
                frames = torch.Tensor(f)
            else:
                frames = torch.cat([frames, f], dim=0)
        return frames

    def __len__(self):
        return len(self.pth)


class UVGDataset(data.Dataset):
    def __init__(self, root):
        self.pth = []
        for rt in sorted(os.listdir(root)):
            self.pth.append(join(root, rt))

    def __getitem__(self, idx):
        nums = 2
        pic_pth = self.pth[idx: idx+nums]
        # start = random.randint(0, 4)
        for i in range(nums):
            f = cv2.imread(pic_pth[i])
            f = torch.Tensor(f).permute(2, 0, 1).unsqueeze(0)
            if i == 0:
                frames = torch.Tensor(f)
            else:
                frames = torch.cat([frames, f], dim=0)
        return frames

    def __len__(self):
        return len(self.pth)


if __name__ == '__main__':
    dataset = VimeoGroupDataset('/data/szy/datasets/vimeo_septuplet/sequences/')
    frame = dataset[0]
    print(frame.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
        self.downscale = nn.AvgPool2d(2)
    def forward(self, x):
        x_down = self.downscale(x)
        x = self.mpconv(x)
        return x + x_down


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x_up = self.upscale(x1)
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # x = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        return x + x_up + x2


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, classes):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        self.up1 = Up(64, 64)
        self.up2 = Up(64, 64)
        self.up3 = Up(64, 64)
        self.outc = OutConv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x

    def predict_recurrent(self, frames):
        '''
        :param frames: [B, T, C, W, H]
        :return: flows [B, T-1, C, W, H]
        '''
        with torch.no_grad():
            for i in range(1, frames.shape[1]):
                if i == 1:
                    flows = self.forward(frames[:, i, ...], frames[:, 0, ...])[0].unsqueeze(0)
                else:
                    flow = self.forward(frames[:, i, ...], frames[:, 0, ...])[0].unsqueeze(0)
                    flows = torch.cat([flows, flow], dim=0)
        return flows.permute(1, 0, 2, 3, 4)
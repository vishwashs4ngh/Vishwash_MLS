# src/models/unet3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pdrop=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout3d(pdrop) if pdrop>0 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if necessary
        if x.shape != skip.shape:
            # compute padding for 5 dims (N,C,D,H,W) -> we need pad for D,H,W
            diffZ = skip.size(2) - x.size(2)
            diffY = skip.size(3) - x.size(3)
            diffX = skip.size(4) - x.size(4)
            pad = [0, diffX, 0, diffY, 0, diffZ]
            x = F.pad(x, pad)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16, pdrop=0.1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base, pdrop)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(base, base*2, pdrop)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(base*2, base*4, pdrop)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base*4, base*8, pdrop)

        self.up3 = UpBlock(base*8, base*4)
        self.up2 = UpBlock(base*4, base*2)
        self.up1 = UpBlock(base*2, base)

        self.out_conv = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        u3 = self.up3(b, e3)
        u2 = self.up2(u3, e2)
        u1 = self.up1(u2, e1)
        out = self.out_conv(u1)
        return out
